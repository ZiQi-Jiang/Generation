import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from einops import rearrange, repeat
from .utils import *


class BottleneckPatchEmbed(nn.Module):
    """
    Patchify image to token. (B, 3, H, W) => (B, num_patches, p*p*3)
    """
    def __init__(self, img_size, patch_size, in_chanels, pca_dim, embed_dim, bias):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size)**2
        
        self.proj1 = nn.Conv2d(in_channels=in_chanels, out_channels=pca_dim, kernel_size=patch_size,stride=patch_size,bias=False) # pca_dim is the bottleneck dimmension
        self.proj2 = nn.Conv2d(in_channels=pca_dim, out_channels=embed_dim,kernel_size=1, stride=1,bias=bias) 
        # in typical vit, only one conv layer, no dimension change.

    def forward(self,x):
        B,C,H,W = x.shape
        x = self.proj2(self.proj1(x)) #(B,p*p*3, p, p)
        x = rearrange(x, 'b c p1 p2 -> b (p1 p2) c') # (B, num_patches, hidden_size)
        return x


# Replace Mlp in sit with SwiGLUFFN
class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size, mlp_hidden_size):
        super().__init__()
        mlp_hidden_size = int(mlp_hidden_size * 2 / 3) # same FLOPs as MLP
        self.w12 = nn.Linear(hidden_size, mlp_hidden_size*2)
        self.w3 = nn.Linear(mlp_hidden_size, hidden_size)
    
    def forward(self, x):
        x1,x2 = self.w12(x).chunk(2, dim=-1)
        x = F.silu(x1) * x2
        x = self.w3(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embed scalar timestep (B,) to vector representations (B, D).
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size,hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size,hidden_size)
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def timestep_embedding(t,frequency_embedding_size, max_period=10000):
        """
        map  t:(B, ) to positional embedding  (B, freqency_embedding_size)
        """
        half = frequency_embedding_size // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32)/half).to(t.device) #(half,)
        args = t[:,None] * freqs[None]  #(B,half)
        embedding = torch.cat([torch.cos(args), torch.sin(args)],dim=-1)
        return embedding
    
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    

class LabelEmbedder(nn.Module):
    """
    Label (B,) to vector (B, D) using dictionary.
    deal with the dropout in train.py, more elegant than sit.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_table = nn.Embedding(num_classes+1, hidden_size)
    
    def forward(self, labels):
        embeddings =  self.embedding_table(labels)
        return embeddings

# actually i think this can be replaced with SwiGLUFFN.
class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size*patch_size*out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size*2)
        )
    
    @torch.compile
    def forward(self, x, c):
        """
        x: (B, N, D)
        c: (B, D)
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1) # (B,N)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class JiTBlock(nn.Module):
    def __init__(self, hidden_size,num_heads,mlp_ration):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.attn = Attention(dim=hidden_size,num_heads=num_heads,qkv_bias=True,qk_norm=True)
        self.mlp = SwiGLUFFN(hidden_size=hidden_size,mlp_hidden_size=int(mlp_ration * hidden_size))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6)
        )
    
    @torch.compile
    def forward(self, x, c, feat_rope=None):
        """
        x: (B, N, D)
        c: (B, D)
        output: (B, N, D)
        """
        # each is (B,D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)

        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x

class JiT(nn.Module):
    def __init__(self, input_size, mlp_ratio, in_channels, num_classes,  depth, hidden_size,  num_heads, bottleneck_dim, in_context_len, in_context_start, patch_size,reg_type=None, reg_depth=4):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes
        self.bottleneck_dim = bottleneck_dim
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.reg_type = reg_type
        self.reg_depth = reg_depth

        # time and condition embedder
        self.t_embedder = TimestepEmbedder(hidden_size=hidden_size)
        self.y_embedder = LabelEmbedder(num_classes=num_classes, hidden_size=hidden_size)

        # patchify: image -> token
        self.x_embedder = BottleneckPatchEmbed(img_size=input_size, patch_size=patch_size, in_chanels=in_channels, pca_dim=bottleneck_dim, embed_dim=hidden_size, bias=True)
        
        # fixed positional embeddings
        self.num_patches = ( input_size // patch_size)**2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        # in context (JiT only)
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1,in_context_len,hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.in_context_posemb,std=0.02)
        
        # rope
        half_rope_dim = hidden_size // num_heads // 2
        seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(half_dim=half_rope_dim, pt_seq_len=seq_len, num_cls_token=0)

        self.feat_rope_incontext = VisionRotaryEmbeddingFast(half_dim=half_rope_dim, pt_seq_len=seq_len, num_cls_token=self.in_context_len)

        #TODO: for JiT-H/G, add proj_drop in the middle block
        self.blocks = nn.ModuleList([
            JiTBlock(hidden_size=hidden_size, num_heads=num_heads, mlp_ration=mlp_ratio) for i in range(depth)
        ])

        assert reg_type in ['repa', 'irepa', 'disperse', 'ig', 'sra', 'none']
        if reg_type == 'repa':
            # self.reg_depth = 8 if  depth >=24 else 4
            self.repa_projector = nn.Sequential(
            nn.Linear(hidden_size, 2048), nn.SiLU(),
            nn.Linear(2048, 2048), nn.SiLU(),
            nn.Linear(2048, 768),
            )
        elif reg_type == 'irepa':
            self.repa_projector = nn.Conv2d(in_channels=hidden_size, out_channels=768, kernel_size=3, stride=1, padding=1)
        elif reg_type == 'disperse':
            self.reg_depth = depth
            pass
        elif reg_type == 'ig': # internal guidance
            self.final_layer_xr = FinalLayer(hidden_size=hidden_size, patch_size=patch_size, out_channels=in_channels)

        elif reg_type == 'sra': # SRA:
            self.final_layer_xr = FinalLayer(hidden_size=hidden_size,patch_size=patch_size, out_channels = in_channels)

        self.final_layer = FinalLayer(hidden_size=hidden_size, patch_size=patch_size, out_channels=in_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
        if self.reg_type == 'ig':
            nn.init.constant_(self.final_layer_xr.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer_xr.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.final_layer_xr.linear.weight, 0)
            nn.init.constant_(self.final_layer_xr.linear.bias, 0)

    def unpatchify(self, x):
        # x: (B, N, D)
        # output: (B, C, H, W)
        h, w = self.input_size // self.patch_size, self.input_size // self.patch_size
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h, w=w, p1 = self.patch_size, p2 = self.patch_size)
        return x

    def forward(self, x, t, y, only_features=False):
        # x: (B, C, H, W)
        # t: (B, )float
        # y: (B,) int
        # output: (B, C, H, W)

        t_emb = self.t_embedder(t) # (B,D)
        y_emb = self.y_embedder(y) # (B,D)
        c = t_emb + y_emb

        x = self.x_embedder(x) # (B,N,D), bottleck layer that will compress information.
        x += self.pos_embed

        for i, block in enumerate(self.blocks):
            if  self.in_context_len > 0 and i == self.in_context_start:
                # decouple the in context tokens into 2 parts: label embed and a residual(pos emb).
                in_context_tokens = repeat(y_emb, 'b d -> b l d', l = self.in_context_len)
                in_context_tokens = in_context_tokens +  self.in_context_posemb 
                x = torch.cat([in_context_tokens, x],dim=1) # (B, N+L, D)
            x = block(x, c, feat_rope=self.feat_rope if i < self.in_context_start else self.feat_rope_incontext)
            
            # repa
            if self.reg_type != "none" and (i + 1) == self.reg_depth:
                x_no_incontext = x[:, self.in_context_len:,:] if i >= self.in_context_start else x # (B, N, D)

                if self.reg_type == 'repa': #linear, direct forward
                    h = self.repa_projector(x_no_incontext) 
                
                elif self.reg_type == 'irepa': # perform spatial normalization
                    H, W = self.input_size // self.patch_size, self.input_size // self.patch_size
                    x_no_incontext = rearrange(x_no_incontext, 'b (h w) d -> b d h w', h=H,w=W)
                    h = self.repa_projector(x_no_incontext)
                    h = rearrange(h, 'b d h w -> b (h w) d')

                    # perform spatial norm
                    gamma = 1.0
                    mean = h.mean(dim=1, keepdim=True)
                    std  = h.std(dim=1, keepdim=True)
                    h = (h - gamma * mean) / (std + 1e-6)

                elif self.reg_type == 'disperse':
                    h = x_no_incontext # (B, N, D)
                elif self.reg_type == 'ig':
                    h = self.final_layer_xr(x_no_incontext, c)
                    h = self.unpatchify(h) # (B, C, H, W)

                elif self.reg_type == 'sra':
                    if self.training: # for stu
                        h = self.final_layer_xr(x_no_incontext, c)
                    else:
                        h = x_no_incontext # for teacher



        x = x[:, self.in_context_len:,:] # back to (B, N, D)
        x = self.final_layer(x, c) # (B, N, D)
        output = self.unpatchify(x) # (B, C, H, W)

        if self.reg_type != "none" and self.training:
            return output, h 
        elif only_features:
            return h
        else:
            return output
    



def JiT_B_16(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=16, **kwargs)


def JiT_B_16_no_incontext(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_len=0, in_context_start=100, patch_size=16, **kwargs)

def JiT_B_32(**kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=32, **kwargs)

def JiT_L_16(**kwargs):
    return JiT(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=16, **kwargs)

def JiT_L_32(**kwargs):
    return JiT(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=32, **kwargs)

def JiT_H_16(**kwargs):
    return JiT(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=16, **kwargs)

def JiT_H_32(**kwargs):
    return JiT(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=32, **kwargs)

JiT_models = {
    'JiT-B/16': JiT_B_16,
    'JiT-B/32': JiT_B_32,
    'JiT-L/16': JiT_L_16,
    'JiT-L/32': JiT_L_32,
    'JiT-H/16': JiT_H_16,
    'JiT-H/32': JiT_H_32,
    'JiT-B/16_no_incontext': JiT_B_16_no_incontext
}