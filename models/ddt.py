import torch
import torch.nn as nn
import math
from .utils import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, pixel_modulate, RMSNorm
from einops import rearrange
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        x_fp32 = x.to(torch.float32)
        rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1,keepdim=True) + self.eps)
        res = x_fp32 * rms
        # res = (res * self.weight).to(x.dtype)
        res = res.to(x.dtype) * self.weight
        return res
    
def precompute_freqs_cis_ex2d(dim, height, width, theta=10000.0, scale=1.0):
    if isinstance(scale, float):
        scale = (scale, scale)
    x_pos = torch.linspace(0, width*scale[0], width)
    y_pos = torch.linspace(0, height*scale[1], height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim)) # Hc/4
    x_freqs = torch.outer(x_pos, freqs).float() # N Hc/4
    y_freqs = torch.outer(y_pos, freqs).float() # N Hc/4
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1) # N,Hc/4,2
    freqs_cis = freqs_cis.reshape(height*width, -1)
    return freqs_cis


def apply_rotary_emb(xq,xk,freqs_cis) :
    freqs_cis = freqs_cis[None, :, None, :]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__( self, dim, num_heads, qkv_bias=False, qk_norm=True, attn_drop=0.0, proj_drop=0.0, norm_layer=RMSNorm) :
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    @torch.compile
    def forward(self, x, pos, mask=None) :
        q, k, v = rearrange(self.qkv(x), 'b n (three heads d) -> three b n heads d', three=3, heads=self.num_heads)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = apply_rotary_emb(q, k, freqs_cis=pos)

        q = rearrange(q, 'b n heads d -> b heads n d')
        k = rearrange(k, 'b n heads d -> b heads n d').contiguous()
        v = rearrange(v, 'b n heads d -> b heads n d').contiguous()

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        x = rearrange(x, 'B num_heads N d -> B N (num_heads d)') # (B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FinalLayer(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(decoder_hidden_dim)
        self.linear = nn.Linear(decoder_hidden_dim, patch_size*patch_size*out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(encoder_hidden_dim, decoder_hidden_dim*2)
        )
    
    @torch.compile
    def forward(self, x, c):
        """
        x: (B, N, D)
        c: (B, D)
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1) # (B,N)
        x = pixel_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    
    
# Same as JiT/DiT, but use bias=False
class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_dim, mlp_hidden_dim):
        super().__init__()
        mlp_hidden_dim = int(mlp_hidden_dim * 2 / 3) # same FLOPs as MLP
        self.w1 = nn.Linear(hidden_dim, mlp_hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, mlp_hidden_dim, bias=False)
        self.w2 = nn.Linear(mlp_hidden_dim, hidden_dim, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# Same as JiT/DiT.
class TimestepEmbedder(nn.Module):
    """
    Embed scalar timestep (B,) to vector representations (B, D).
    """
    def __init__(self, hidden_dim, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size,hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim,hidden_dim)
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def timestep_embedding(t,frequency_embedding_size, max_period=10):
        """
        map  t:(B, ) to positional embedding  (B, freqency_embedding_size)
        """
        half = frequency_embedding_size // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half) #(half,)
        args = t[:,None] * freqs[None]  #(B,half)
        embedding = torch.cat([torch.cos(args), torch.sin(args)],dim=-1)
        return embedding
    
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    


# same as JiT/DiT.
class LabelEmbedder(nn.Module):
    """
    Label (B,) to vector (B, D) using dictionary.
    deal with the dropout in train.py, more elegant than sit.
    """
    def __init__(self, num_classes, hidden_dim):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_table = nn.Embedding(num_classes+1, hidden_dim)
    
    def forward(self, labels):
        embeddings =  self.embedding_table(labels)
        return embeddings
    

# Lightning DiT. Use advanced RMSNorm, RoPE, SwiGU. qkv_bias = False.
class DiTBlock(nn.Module):
    def __init__(self, hidden_dim,num_heads,mlp_ration):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)
        self.attn = Attention(dim=hidden_dim,num_heads=num_heads,qkv_bias=False,qk_norm=True)
        self.mlp = SwiGLUFFN(hidden_dim=hidden_dim,mlp_hidden_dim=int(mlp_ration * hidden_dim))
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 6)
        )

    @torch.compile
    def forward(self, x, c, feat_rope=None):
        """
        x: (B, N, D)
        c: (B, 1, D)
        output: (B, N, D)
        """
        # each is (B,1, D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)

        x = x + gate_msa * self.attn(pixel_modulate(self.norm1(x), shift_msa, scale_msa), pos=feat_rope, mask=None)
        x = x + gate_mlp * self.mlp(pixel_modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x

class DDTBlock(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, num_heads,mlp_ration):
        super().__init__()
        self.norm1 = RMSNorm(decoder_hidden_dim)
        self.norm2 = RMSNorm(decoder_hidden_dim)
        self.attn = Attention(dim=decoder_hidden_dim,num_heads=num_heads,qkv_bias=False,qk_norm=True)
        self.mlp = SwiGLUFFN(hidden_dim=decoder_hidden_dim,mlp_hidden_dim=int(mlp_ration * decoder_hidden_dim))
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(encoder_hidden_dim, decoder_hidden_dim * 6)
        )

    @torch.compile
    def forward(self, x, s, feat_rope=None):
        """
        x: (B, N, D)
        c: (B, 1, D)
        output: (B, N, D)
        """
        # each is (B,1, D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(s).chunk(6, dim=-1)

        x = x + gate_msa * self.attn(pixel_modulate(self.norm1(x), shift_msa, scale_msa), pos=feat_rope, mask=None)
        x = x + gate_mlp * self.mlp(pixel_modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x

class DDT(nn.Module):
        def __init__(self, input_size, in_channels, mlp_ratio,  num_classes,  encoder_depth, decoder_depth, encoder_hidden_dim,  num_heads, decoder_hidden_dim,  patch_size,):
            """
            in_channel: 3 for RBG
            mlp_ratio: 4.0 default. hidden size in mlp.
            num_classes: 1000 default, number of classes in the dataset.
            encoder_depth: num of blocks for Dit Blocks(lightning DiT), preprocesing patches in Enocder.
            decoder_depth: num of blocks for DDT blocks(prposed in this PixelDiT), pixel-level operation in Decoder.
            hidden_dim:
            num_heads: in attention.
            pixel_hidden_dim:
            patch_size: 
            """
            super().__init__()
            self.input_size = input_size
            self.in_channels = in_channels
            self.out_channels = in_channels
            self.mlp_ratio = mlp_ratio
            self.num_classes = num_classes
            self.encoder_depth = encoder_depth
            self.decoder_depth = decoder_depth
            self.encoder_hidden_dim = encoder_hidden_dim
            self.decoder_hidden_size = decoder_hidden_dim
            self.num_heads = num_heads
            self.patch_size = patch_size


            # self.pos_embed = nn.Parameter(torch.zeros(1, self.input_size*self.input_size, pixel_hidden_dim), requires_grad=False)
            # self.register_buffer('dpos_embed', torch.zeros(1, self.input_size*self.input_size, decoder_hidden_dim), persistent=False)  # (1, H*W, d)
            # Initialize (and freeze) pos_embed by sin-cos embedding:
            # pixel_pos_embed = get_2d_sincos_pos_embed(self.pixel_hidden_dim, self.input_size)
            # self.pixel_pos_embed.data.copy_(torch.from_numpy(pixel_pos_embed).float().unsqueeze(0))

            # time and condition embedder
            self.x_embedder = nn.Linear(in_channels*patch_size*patch_size, decoder_hidden_dim) # pixel embedding, for decoder.
            self.s_embedder = nn.Linear(in_channels*patch_size*patch_size, encoder_hidden_dim) # patch embedding, for encoder.
            self.t_embedder = TimestepEmbedder(hidden_dim=encoder_hidden_dim)
            self.y_embedder = LabelEmbedder(num_classes=num_classes, hidden_dim=encoder_hidden_dim)

            self.final_layer = FinalLayer(encoder_hidden_dim=encoder_hidden_dim, decoder_hidden_dim=decoder_hidden_dim, patch_size=patch_size, out_channels=in_channels)

             # rope
            seq_len = input_size // patch_size
            # self.feat_rope = VisionRotaryEmbeddingFast(half_dim=half_rope_dim, pt_seq_len=seq_len, num_cls_token=0)
            self.register_buffer("encoder_feat_rope", precompute_freqs_cis_ex2d(encoder_hidden_dim // num_heads, seq_len, seq_len), persistent=False)
            self.register_buffer("decoder_feat_rope", precompute_freqs_cis_ex2d(decoder_hidden_dim // num_heads, seq_len, seq_len), persistent=False)

            self.encoder_blocks = nn.ModuleList([DiTBlock(hidden_dim=encoder_hidden_dim, num_heads=num_heads, mlp_ration=mlp_ratio) for _ in range(encoder_depth)])

            self.decoder_blocks = nn.ModuleList([DDTBlock(encoder_hidden_dim=encoder_hidden_dim, num_heads=num_heads, mlp_ration=mlp_ratio, decoder_hidden_dim=decoder_hidden_dim) for _ in range(decoder_depth)])

            self.DDT_initialize_weights()

        def DDT_initialize_weights(self):
            # Initialize transformer layers:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)

            for embedder in (self.x_embedder, self.s_embedder):
                nn.init.xavier_uniform_(embedder.weight)
                nn.init.zeros_(embedder.bias)

            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
            nn.init.zeros_(self.final_layer.linear.weight)
            nn.init.zeros_(self.final_layer.linear.bias)
            for block in self.encoder_blocks:
                nn.init.zeros_(block.adaLN_modulation[0].weight)
                nn.init.zeros_(block.adaLN_modulation[0].bias)
            for block in self.decoder_blocks:
                nn.init.zeros_(block.adaLN_modulation[0].weight)
                nn.init.zeros_(block.adaLN_modulation[0].bias)

        def forward(self, x, t, y, s=None):
            """
            x: (B, C, H, W)
            t: (B,)
            y: (B,)
            s: (B*N,D), output of encoder. 
            """
            B,C,H,W = x.shape

            x_patch = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=self.patch_size, p2=self.patch_size)  # (B, N, patch_size*patch_size*C)
            t_emb = self.t_embedder(t).unsqueeze(1)  # (B, 1, D_e)
            y_emb = self.y_embedder(y).unsqueeze(1)  # (B, 1, D_e)

            if s is None: #DiT Encoder part : x_patch -> s
                # typically, dit combine pathify and linear projection into Conv. PixelDiT sepereate them into two steps.
                s = self.s_embedder(x_patch)  # (B, N, D)
                c = F.silu(t_emb + y_emb)
                for i in range(self.encoder_depth):
                    s = self.encoder_blocks[i](s, c, feat_rope=self.encoder_feat_rope)  # (B, N, D)

                s = F.silu(s + t_emb ) # (B, N, D)

            x = self.x_embedder(x_patch) # (B, N, D_d)

            for i in range(self.decoder_depth): # DDT Decoder
                x = self.decoder_blocks[i](x, s, feat_rope=self.decoder_feat_rope)  # (B*N, patch_size*patch_size, d)
            out = self.final_layer(x, s)  # (B, N, p1*p2*c)
            out  = rearrange(out, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)', b=B, h=H//self.patch_size, w=W//self.patch_size, p1=self.patch_size, p2=self.patch_size)  # (B, C, H, W)

            return out



def DDT_B_16(**kwargs):
    return DDT(encoder_depth=12, decoder_depth=2, encoder_hidden_dim=768, num_heads=12, decoder_hidden_dim=768, patch_size=16, **kwargs)

def myDDT_B_16(**kwargs):
    return DDT(encoder_depth=10, decoder_depth=2, encoder_hidden_dim=768, num_heads=12, decoder_hidden_dim=768, patch_size=16, **kwargs)

def DDT_B_4(**kwargs):
    return DDT(encoder_depth=12, decoder_depth=2, encoder_hidden_dim=768, num_heads=12, decoder_hidden_dim=768, patch_size=4,  **kwargs)

def DDT_B_32(**kwargs):
    return DDT(encoder_depth=12, decoder_depth=2, encoder_hidden_dim=768, num_heads=12, decoder_hidden_dim=768, patch_size=32,  **kwargs)

def DDT_L_16(**kwargs):
    return DDT(encoder_depth=22, decoder_depth=4, encoder_hidden_dim=1152, num_heads=16, decoder_hidden_dim=1024, patch_size=16,  **kwargs)

def DDT_XL_16(**kwargs):
    return DDT(encoder_depth=26, decoder_depth=4, encoder_hidden_dim=1152, num_heads=16, decoder_hidden_dim=1024, patch_size=16,  **kwargs)



DDT_models = {
    "DDT-B/16": DDT_B_16,
    "myDDT-B/16": myDDT_B_16,
    "DDT-B/4": DDT_B_4,
    "DDT-B/32": DDT_B_32,
    "DDT-L/16": DDT_L_16,
    "DDT-XL/16": DDT_XL_16
}