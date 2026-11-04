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


class PiTFinalLayer(nn.Module):
    def __init__(self, hidden_dim, out_channels):
        super().__init__()
        self.norm = RMSNorm(hidden_dim, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, out_channels, bias=True)

    @torch.compile
    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x

class PiTMLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
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

class PiTBlock(nn.Module):
    """
    Core Component of PixelDiT. 
    """
    def __init__(self, hidden_dim, num_heads, mlp_ration, pixel_hidden_dim, patch_size ):
        super().__init__()
        self.attn_dim = hidden_dim # we fix attn dim as hidden dim.
        self.patch_size = patch_size
        # patch-level attention.
        self.compress_to_attn = nn.Linear(patch_size * patch_size * pixel_hidden_dim, self.attn_dim)
        self.expand_from_attn = nn.Linear(self.attn_dim, patch_size * patch_size * pixel_hidden_dim)

        self.norm1 = RMSNorm(pixel_hidden_dim)
        self.attn = Attention(dim=self.attn_dim, num_heads=num_heads, qkv_bias=False, qk_norm=True)
        self.norm2 = RMSNorm(pixel_hidden_dim)

        self.mlp = PiTMLP(dim=pixel_hidden_dim, mlp_ratio=mlp_ration)

        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_dim, pixel_hidden_dim * 6 * patch_size * patch_size) # pixel-level control
        )
        
    @torch.compile
    def forward(self, x_pixel, s, feat_rope=None):
        """
        x_pixel: (BN, patch_size*patch_size, d)
        s: (B,N,D)
        output: (B*N, patch_size*patch_size, d)
        """
        BN, P2, d = x_pixel.shape
        B, N, D = s.shape
        # pixel-specific control from patch-level representation s.
        s_pixel = rearrange(s, 'b n d -> (b n)  d')  # (B*N,  D) for pixel-level decoder.
        s_condition = self.adaLN_modulation(s_pixel) # (B*N, d*6*patch_size*patch_size)
        s_condition = rearrange(s_condition, 'bn (p2 d6) -> bn p2 d6', p2=P2)  # (B*N, patch_size*patch_size, 6*d)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = s_condition.chunk(6, dim=-1)  # (BN,p2,d) each


        # attn part: norm1, modulate, attn(compress, attn expand in PiT)
        x_norm = pixel_modulate(self.norm1(x_pixel), shift_msa, scale_msa)  # (B*N, P2, d)
        x_patch = rearrange(x_norm, '(b n) p2 d -> b n (p2 d)', n=N)  # (B, N, patch_size*patch_size*d)
        x_comp = self.compress_to_attn(x_patch)  # (B, N, attn_dim)
        x_comp = self.attn(x_comp, pos=feat_rope, mask=None)  # (B, N, attn_dim)
        x_patch = self.expand_from_attn(x_comp)  # (B, N, patch_size*patch_size*d)
        res = rearrange(x_patch, 'b n (p2 d) -> (b n) p2 d', b=B, n=N, p2=P2, d=d)  # (B*N, P2, d)
        x_pixel = x_pixel + gate_msa * res # (B*N, P2, d)

        #mlp part: norm2, modulate, mlp
        x_pixel = x_pixel + gate_mlp * self.mlp(pixel_modulate(self.norm2(x_pixel), shift_mlp, scale_mlp))  # (B*N, P2, d)

        return x_pixel


class PiT(nn.Module):
        def __init__(self, input_size, in_channels, mlp_ratio,  num_classes,  patch_depth, pixel_depth, hidden_dim,  num_heads, pixel_hidden_dim,  patch_size,):
            """
            in_channel: 3 for RBG
            mlp_ratio: 4.0 default. hidden size in mlp.
            num_classes: 1000 default, number of classes in the dataset.
            patch_depth: num of blocks for Dit Blocks(lightning DiT), preprocesing patches in Enocder.
            pixel_depth: num of blocks for PiT blocks(prposed in this PixelDiT), pixel-level operation in Decoder.
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
            self.patch_depth = patch_depth
            self.pixel_depth = pixel_depth
            self.hidden_dim = hidden_dim
            self.hidden_size = hidden_dim
            self.num_heads = num_heads
            self.pixel_hidden_dim = pixel_hidden_dim
            self.patch_size = patch_size


            # self.pixel_pos_embed = nn.Parameter(torch.zeros(1, self.input_size*self.input_size, pixel_hidden_dim), requires_grad=False)
            self.register_buffer('pixel_pos_embed', torch.zeros(1, self.input_size*self.input_size, pixel_hidden_dim), persistent=False)  # (1, H*W, d)
            # Initialize (and freeze) pos_embed by sin-cos embedding:
            pixel_pos_embed = get_2d_sincos_pos_embed(self.pixel_hidden_dim, self.input_size)
            self.pixel_pos_embed.data.copy_(torch.from_numpy(pixel_pos_embed).float().unsqueeze(0))

            # time and condition embedder
            self.x_embedder = nn.Linear(in_channels, pixel_hidden_dim) # pixel embedding, for decoder.
            self.s_embedder = nn.Linear(in_channels*patch_size*patch_size, hidden_dim) # patch embedding, for encoder.
            self.t_embedder = TimestepEmbedder(hidden_dim=hidden_dim)
            self.y_embedder = LabelEmbedder(num_classes=num_classes, hidden_dim=hidden_dim)

            self.final_layer = PiTFinalLayer(hidden_dim=pixel_hidden_dim, out_channels=in_channels)

             # rope
            seq_len = input_size // patch_size
            # self.feat_rope = VisionRotaryEmbeddingFast(half_dim=half_rope_dim, pt_seq_len=seq_len, num_cls_token=0)
            self.register_buffer("feat_rope", precompute_freqs_cis_ex2d(hidden_dim // num_heads, seq_len, seq_len), persistent=False)

            self.patch_blocks = nn.ModuleList([DiTBlock(hidden_dim=hidden_dim, num_heads=num_heads, mlp_ration=mlp_ratio) for _ in range(patch_depth)])

            self.pixel_blocks = nn.ModuleList([PiTBlock(hidden_dim=hidden_dim, num_heads=num_heads, mlp_ration=mlp_ratio, pixel_hidden_dim=pixel_hidden_dim, patch_size=patch_size) for _ in range(pixel_depth)])

            self.pit_initialize_weights()

        def pit_initialize_weights(self):


            w = self.s_embedder.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.s_embedder.bias, 0)
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
            nn.init.zeros_(self.final_layer.linear.weight)
            nn.init.zeros_(self.final_layer.linear.bias)
            for block in self.patch_blocks:
                nn.init.zeros_(block.adaLN_modulation[0].weight)
                nn.init.zeros_(block.adaLN_modulation[0].bias)
            for block in self.pixel_blocks:
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
            t_emb = self.t_embedder(t).unsqueeze(1)  # (B, 1, D)
            y_emb = self.y_embedder(y).unsqueeze(1)  # (B, 1, D)

            if s is None: #DiT Encoder part : x_patch -> s
                # typically, dit combine pathify and linear projection into Conv. PixelDiT sepereate them into two steps.
                s = self.s_embedder(x_patch)  # (B, N, D)
                c = F.silu(t_emb + y_emb)
                for i in range(self.patch_depth):
                    s = self.patch_blocks[i](s, c, feat_rope=self.feat_rope)  # (B, N, D)

                s = F.silu(s + t_emb) # (B, N, D)
            N, D = s.shape[1], s.shape[2]

            # pachify x
            x_pixel = rearrange(x, 'b c h w -> b h w c').contiguous()  # (B, H, W, C)
            pixel_pos_embed = rearrange(self.pixel_pos_embed, '1 (h w) d -> 1 h w d', h=H, w=W)  # (1, H, W, d)
            x_pixel = self.x_embedder(x_pixel)  # (B, H, W, d)

            x_pixel = x_pixel + pixel_pos_embed
            x_pixel = rearrange(x_pixel, 'b (h p1) (w p2) d -> (b h w) (p1 p2) d', p1=self.patch_size, p2=self.patch_size)  # (B*N, patch_size*patch_size, d)

            for i in range(self.pixel_depth): # PiT Decoder
                # no interaction between different patches,
                x_pixel = self.pixel_blocks[i](x_pixel, s, feat_rope=self.feat_rope)  # (B*N, patch_size*patch_size, d)
            out = self.final_layer(x_pixel)  # (B*N, patch_size*patch_size, C)
            out  = rearrange(out, '(b h w) (p1 p2) c -> b c (h p1) (w p2)', b=B, h=H//self.patch_size, w=W//self.patch_size, p1=self.patch_size, p2=self.patch_size)  # (B, C, H, W)

            return out



def PiT_B_16(**kwargs):
    return PiT(patch_depth=12, pixel_depth=2, hidden_dim=768, num_heads=12, pixel_hidden_dim=16, patch_size=16, **kwargs)

def PiT_B_4(**kwargs):
    return PiT(patch_depth=12, pixel_depth=2, hidden_dim=768, num_heads=12, pixel_hidden_dim=16, patch_size=4,  **kwargs)

def PiT_B_32(**kwargs):
    return PiT(patch_depth=12, pixel_depth=2, hidden_dim=768, num_heads=12, pixel_hidden_dim=16, patch_size=32,  **kwargs)

def PiT_L_16(**kwargs):
    return PiT(patch_depth=22, pixel_depth=4, hidden_dim=1024, num_heads=16, pixel_hidden_dim=16, patch_size=16,  **kwargs)

def PiT_XL_16(**kwargs):
    return PiT(patch_depth=26, pixel_depth=4, hidden_dim=1152, num_heads=16, pixel_hidden_dim=16, patch_size=16,  **kwargs)



PiT_models = {
    "PiT-B/16": PiT_B_16,
    "PiT-B/4": PiT_B_4,
    "PiT-B/32": PiT_B_32,
    "PiT-L/16": PiT_L_16,
    "PiT-XL/16": PiT_XL_16
}