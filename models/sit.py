# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from collections import OrderedDict

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        # t: (B,) 
        # t_emb: (B, hidden_size)
        t_scaled = t.float() * 1000  # scale to [0, 1000], a design flaw in SiT.
        t_freq = self.timestep_embedding(t_scaled, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        # dropout_prob: whether need unconditional generation (cfg) support, defaut 0.1
        use_cfg = dropout_prob > 0 
        self.num_classes = num_classes
        self.embedding_table = nn.Embedding(num_classes + use_cfg, hidden_size)
        self.dropout, self.hidden_size = dropout_prob, hidden_size 
        

    def forward(self, labels, train):
        # labels: (B,)
        use_cfg = self.dropout > 0
        if use_cfg and train:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout
            labels = torch.where(drop_ids, self.num_classes, labels)
        embeddings = self.embedding_table(labels)
        return embeddings


class SiTBlock(nn.Module):

    def __init__(self, hidden_size, num_heads,mlp_ration, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.attn = Attention(hidden_size, num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        mlp_hidden_size = int(hidden_size * mlp_ration)
        approx_gelu = lambda: nn.GELU(approximate='tanh')
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_size, act_layer=approx_gelu)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size , hidden_size * 6, bias=True)
        )

    
    def forward(self, x, c):
        # x: (B, N, D), c: (B, D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x



class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        out_dim = patch_size * patch_size * out_channels
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(hidden_size , hidden_size * 2, bias=True)
        )


    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)

        x = modulate(self.norm(x), shift, scale)
    
        x = self.linear(x)
        return x

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class SiT(nn.Module):
    def __init__(
            self,
            input_size=32,
            patch_size=2,
            in_channels=4,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            num_classes=1000,
            learn_sigma=True, # consistent with DiT, but actually useless in DiT.
            use_repa=False, 
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels*2 if learn_sigma else in_channels 
        self.patch_size = patch_size
        self.num_head = num_heads
        self.use_repa = use_repa
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        num_patches = (input_size // patch_size) ** 2

        # fixed 2d cosine positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])

     

        # for REPA: XL/L, depth is 8; B/S is 4.
        # REPA dim: 2048 for hiddent, 768 as output, same as DiN0v2-vit-b.
        if use_repa:
            self.repa_depth = 8 if  depth >=24 else 4
            self.repa_projector = nn.Sequential(
            nn.Linear(hidden_size, 2048), nn.SiLU(),
            nn.Linear(2048, 2048), nn.SiLU(),
            nn.Linear(2048, 768),
        )
            
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
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
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        # x: (N, num_patches, patch_size*patch_size*out_channels)
        # imgs: (N, out_channels, H, W, ) 
        p = self.patch_size
        c = self.out_channels
        h = w = int(math.sqrt(x.shape[1]))
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs


    
    def forward(self, x, t, y):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, num_patches, hidden_size)
        N,T,D = x.shape
        t = self.t_embedder(t)  # (N, hidden_size)
        # only drop labels during training
        y = self.y_embedder(y, self.training)  # (N, hidden_size)

        c = t + y  # (N, hidden_size)

        for i,block in enumerate(self.blocks):
            x = block(x, c) # [N, num_patches, hidden_size]
            if self.training and self.use_repa and (i + 1) == self.repa_depth:
                h = self.repa_projector(x.reshape(-1, D)) 
                h = h.reshape(N, T, -1) # [N, num_patches, 768]
        x = self.final_layer(x, c)  # (N, num_patches, patch_size*patch_size*out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)  
        
        if self.use_repa and self.training:
            return x, h
        else:
            return x


    def forward_with_cfg(self, x, t, y, cfg_scale):
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        # x: (2*B, C, H,W)
        # t: (2*B,)
        # y: (2*B,) 
        cond_half = x[: len(x)//2]
        # for cfg, the input should be the same, 
        combined_input = torch.cat([cond_half, cond_half], dim=0)
        model_output = self.forward(combined_input, t, y) # (2*B, C, H, W)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_output[:, :self.in_channels], model_output[:, self.in_channels:]
        # eps, rest = model_output[:, :3], model_output[:, 3:] #[2*B, 3, H, W], [2*B, 1, H, W]
        cond_eps, uncond_eps = eps.chunk(2, dim=0) # [B, 3, H, W], [B, 3, H, W]
        guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps) #[B, 3, H, W]
        eps = torch.cat([guided_eps, guided_eps], dim=0) #[2*B, 3, H, W], sceond one as placeholder
        out = torch.cat([eps, rest], dim=1) #[2*B, C, H, W]
        return out

#################################################################################
#                                   SiT Configs                                  #
#################################################################################

def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def SiT_XL_4(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def SiT_XL_8(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def SiT_L_2(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def SiT_L_4(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def SiT_L_8(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def SiT_B_2(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def SiT_B_4(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def SiT_B_8(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def SiT_S_2(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def SiT_S_4(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def SiT_S_8(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


SiT_models = {
    'SiT-XL/2': SiT_XL_2,  'SiT-XL/4': SiT_XL_4,  'SiT-XL/8': SiT_XL_8,
    'SiT-L/2':  SiT_L_2,   'SiT-L/4':  SiT_L_4,   'SiT-L/8':  SiT_L_8,
    'SiT-B/2':  SiT_B_2,   'SiT-B/4':  SiT_B_4,   'SiT-B/8':  SiT_B_8,
    'SiT-S/2':  SiT_S_2,   'SiT-S/4':  SiT_S_4,   'SiT-S/8':  SiT_S_8,
}
