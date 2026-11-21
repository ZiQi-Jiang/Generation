import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from einops import rearrange, repeat
from .utils import *
from .jit_clean import JiT
from .dino_vit import DINOHead
from .utils import VisionRotaryEmbeddingFast


class DinoJiT(JiT):
    def __init__(self, *args, reg_depth=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_depth = reg_depth
        dim = self.hidden_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = DINOHead(dim)
        for layer in self.head.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)

        self.repa_projector = nn.Sequential(
            nn.Linear(dim, 2048), nn.SiLU(),
            nn.Linear(2048, 2048), nn.SiLU(),
            nn.Linear(2048, dim),
        )
        half_dim = dim // self.num_heads // 2
        self.dino_rope = nn.ModuleDict({
            str(size): VisionRotaryEmbeddingFast(
                half_dim=half_dim,
                pt_seq_len=size // self.patch_size,
                num_cls_token=1,
            )
            for size in (96, 224, self.input_size)
        })

    def interpolate_pos_emb(self, x):
        h, w = x.shape[-2:]
        if (h, w) == (self.input_size, self.input_size):
            return self.pos_embed
        grid = self.input_size // self.patch_size
        pos = rearrange(self.pos_embed, "1 (h w) d -> 1 d h w", h=grid, w=grid)
        pos = F.interpolate(
            pos, size=(h // self.patch_size, w // self.patch_size),
            mode="bicubic", align_corners=False,
        )
        return rearrange(pos, "1 d h w -> 1 (h w) d")

    def dino_tokens(self, images):
        batch, _, size, _ = images.shape
        t = images.new_ones(batch)
        y = torch.full(
            (batch,), self.num_classes, device=images.device, dtype=torch.long,
        )
        c = self.t_embedder(t) + self.y_embedder(y)

        x = self.x_embedder(images)
        x = x + self.interpolate_pos_emb(images).to(x.dtype)
        cls = self.cls_token.to(x.dtype).expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)

        rope = self.dino_rope[str(size)]
        for i in range(self.reg_depth):
            x = self.blocks[i](x, c, feat_rope=rope)
        return x

    def dino_forward(self, images):
        return self.head(self.norm(self.dino_tokens(images)[:, 0]))

    def forward(self, x, t, y, global_views, local_views):
        cache = []

        def save_features(module, inputs, output):
            cache.append(output)

        with self.blocks[self.reg_depth - 1].register_forward_hook(save_features):
            x1_pred = super().forward(x, t, y)

        h = self.repa_projector(cache[0][:, -self.num_patches:])
        stu_logits = torch.cat([
            self.dino_forward(global_views),
            self.dino_forward(local_views),
        ], dim=0)
        return x1_pred, h, stu_logits

def DinoJiT_S_16(**kwargs):
    return DinoJiT(depth=12, hidden_size=384, num_heads=6, 
                bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=16, **kwargs)

def DinoJiT_S_16_no_incontext(**kwargs):
    return DinoJiT(depth=12, hidden_size=384, num_heads=6, 
                bottleneck_dim=128, in_context_len=0, in_context_start=100, patch_size=16, **kwargs)

def DinoJiT_B_16(**kwargs):
    return DinoJiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=16, **kwargs)


def DinoJiT_B_16_no_incontext(**kwargs):
    return DinoJiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_len=0, in_context_start=100, patch_size=16, **kwargs)

def DinoJiT_B_32(**kwargs):
    return DinoJiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=32, **kwargs)

def DinoJiT_L_16(**kwargs):
    return DinoJiT(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=16, **kwargs)

def DinoJiT_L_32(**kwargs):
    return DinoJiT(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=32, **kwargs)

def DinoJiT_H_16(**kwargs):
    return DinoJiT(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=16, **kwargs)

def DinoJiT_H_32(**kwargs):
    return DinoJiT(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=32, **kwargs)

DinoJiT_models = {
    'DinoJiT-S/16': DinoJiT_S_16,
    'DinoJiT-B/16': DinoJiT_B_16,
    'DinoJiT-B/32': DinoJiT_B_32,
    'DinoJiT-L/16': DinoJiT_L_16,
    'DinoJiT-L/32': DinoJiT_L_32,
    'DinoJiT-H/16': DinoJiT_H_16,
    'DinoJiT-H/32': DinoJiT_H_32,
    'DinoJiT-B/16_no_incontext': DinoJiT_B_16_no_incontext,
    'DinoJiT-S/16_no_incontext': DinoJiT_S_16_no_incontext
}


        


