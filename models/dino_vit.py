import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat, einsum

class DropPath(nn.Module):

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor

class PatchEmbed(nn.Module):
    # [B, C, H, W] -> [B, N, D]
    def __init__(self,  patch_size, in_channels, dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=dim, kernel_size= patch_size, stride= patch_size) # [B, C, H, W] -> [B, D, p, p]
    def forward(self, x):
        x = self.conv(x) #[B,D, p, p]
        x = rearrange(x, 'B D p1 p2 -> B (p1 p2) D')
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias, qk_norm):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # modern QK-Norm:
        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6) if qk_norm else nn.Identity()

    def forward(self, x):
        B, N, D = x.shape
        q,k, v = rearrange(self.qkv(x), ' B N (three h d) -> three B h N d', three = 3, h = self.num_heads)

        q,k = self.q_norm(q), self.k_norm(k)

        
        x = F.scaled_dot_product_attention(q, k, v)
        # attn = (q @ k.transpose(-1, -2)) * self.scale # [B, h, N, N]
        # attn = attn.softmax(dim=-1)
        # x = attn @ v # [B, h, N, D]

        x = rearrange(x, 'B h N D -> B N (h D)')
        x = self.proj(x)

        return x


class Mlp(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, in_dim)
    
    def forward(self,x):
        return self.fc2(self.act(self.fc1(x)))


class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias, qk_norm,drop_prob):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = Mlp(in_dim = dim, hidden_dim = int(mlp_ratio * dim))
        self.drop_path = DropPath(drop_prob= drop_prob)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, dim=384, depth=12, num_heads=6, mlp_ratio=4.0, drop_path_rate=0.1, qkv_bias=True, qk_norm=False):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.dim = dim
        self.num_patches = (img_size // patch_size) **2

        self.patch_emb = PatchEmbed(patch_size=patch_size, in_channels=in_channels, dim=dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            ViTBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, drop_prob=dpr[i]) for i in range(depth) 
            )
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = DINOHead(in_dim=dim)
        self.init_weights()
        
    def init_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.apply(_basic_init) # For LayerNorm, default: weight=1 bias=0

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def interpolate_pos_emb(self, x):
        B, c, h, w = x.shape
        H =  W =  int(math.sqrt(self.num_patches))
        if h == self.img_size and w == self.img_size:
            return self.pos_embed
        cls_emb, patch_emb = self.pos_embed[:,:1], self.pos_embed[:,1:]
        patch_emb = rearrange(patch_emb, '1 (H W) D -> 1 D H W', H=H, W=W)
        w0, h0 = w // self.patch_size + 0.1, h // self.patch_size + 0.1
        patch_emb = F.interpolate(patch_emb, scale_factor=(h0/H, w0/W), mode='bicubic', align_corners=False)
        patch_emb = rearrange(patch_emb, '1 D h w -> 1 (h w) D')
        pos_embed = torch.cat([cls_emb, patch_emb], dim=1)
        return pos_embed



    def forward_feature(self, x):
        pos_embed = self.interpolate_pos_emb(x)
        x = self.patch_emb(x) #[B,C,H,W] -> [B,N,D]

        cls_tokes = repeat(self.cls_token, '1 1 d -> B 1 d', B = x.shape[0])
        x = torch.cat([cls_tokes,x],dim=1)

        x = x + pos_embed
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self,x):
        x = self.forward_feature(x)
        out = self.head(self.norm(x[:,0])) #[B,1, num_classes]
        return out


class DINOHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, 256),
        )
        self.prototypes = nn.Parameter(torch.empty(65536, 256)) # [K, D]
        # nn.init.trunc_normal_(self.prototypes, std=0.02)
        nn.init.kaiming_uniform_(self.prototypes, a=math.sqrt(5))
        self.prototype_scale = nn.Parameter(torch.ones(65536, 1))
    
    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2) #[B, D]
        # prototypes = F.normalize(self.prototypes, dim=-1, p=2) #[K,D]
        prototypes = F.normalize(self.prototypes, dim=-1, p=2) * self.prototype_scale #[K,D]

        # logits = einsum( 'B  D, num_prototypes D -> B num_prototypes', x, prototypes)
        logits = x @ prototypes.T
        return logits
    
    
    
        


