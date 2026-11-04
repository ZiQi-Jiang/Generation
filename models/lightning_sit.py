# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed
from collections import OrderedDict
from einops import rearrange
from .utils import *
from .jit_reg import SwiGLUFFN, SRAHead, TimestepEmbedder, LabelEmbedder, FinalLayer





class SiTBlock(nn.Module):

    def __init__(self, hidden_size, num_heads,mlp_ration, **block_kwargs):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(dim=hidden_size,num_heads=num_heads,qkv_bias=True,qk_norm=True)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size=hidden_size,mlp_hidden_size=int(mlp_ration * hidden_size))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size , hidden_size * 6,)
        )

    # @torch.compile 
    @torch.compile(dynamic=False)
    def forward(self, x, c, rope=None):
        # x: (B, N, D), c: (B, D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


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
            num_classes=1000,
            learn_sigma=False, # consistent with DiT, but actually useless in DiT.
            reg_type='none', # 'repa', 'irepa', 'disperse', 'ig', 'sra', 'none',
            reg_depth=4,
            reg_mask_ratio=0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels*2 if learn_sigma else in_channels 
        self.patch_size = patch_size
        self.num_head = num_heads
        self.reg_type = reg_type
        self.reg_depth = reg_depth
        self.reg_mask_ratio= reg_mask_ratio
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        num_patches = (input_size // patch_size) ** 2

        # fixed 2d cosine positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)


         # rope
        half_rope_dim = hidden_size // num_heads // 2
        seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(half_dim=half_rope_dim, pt_seq_len=seq_len, num_cls_token=0)


        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])

        # for REPA: XL/L, depth is 8; B/S is 4.
        # REPA dim: 2048 for hiddent, 768 as output, same as DiN0v2-vit-b.
        assert reg_type in ['repa', 'irepa', 'disperse', 'ig', 'sra', 'none', 'sd-dit']
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
            pass
        elif reg_type == 'ig': # internal guidance
            self.final_layer_xr = FinalLayer(hidden_size=hidden_size, patch_size=patch_size, out_channels=in_channels)

        elif reg_type == 'sra': # SRA:
            # self.final_layer_xr = FinalLayer(hidden_size=hidden_size,patch_size=patch_size, out_channels = in_channels)
            self.final_layer_xr = SRAHead(in_dim=hidden_size, out_dim=hidden_size)

        elif reg_type == 'sd-dit': # SD-DiT:
            self.K = 8192 # dino, number of prototype.
            self.register_buffer("center_patch", torch.zeros(1,1,self.K)) # Centering in DINO
            self.m_p = 0.9 # moving average of patch centern
            self.sddit_projector = DINOPatchHead(hidden_size, self.K)
            self.feat_rope= None # Currently, we do not use ROPE

            
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
    def forward(self, x, t, y, depth=None):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        if depth is not None:
            return self.forward_with_features(x, t, y, depth)
        
        x = self.x_embedder(x) + self.pos_embed  # (N, num_patches, hidden_size)
        N,T,D = x.shape
        t = self.t_embedder(t)  # (N, hidden_size)
        # only drop labels during training
        y = self.y_embedder(y)  # (N, hidden_size)

        c = t + y  # (N, hidden_size)

        for i,block in enumerate(self.blocks):
            x = block(x, c, rope=self.feat_rope) # [N, num_patches, hidden_size]
        x = self.final_layer(x, c)  # (N, num_patches, patch_size*patch_size*out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        return x

    
    def forward_with_features(self, x, t, y, depth=None):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        is_student = (self.training == True)
    
        x = self.x_embedder(x) + self.pos_embed  # (N, num_patches, hidden_size)
        N,T,D = x.shape

        if self.reg_type == 'sd-dit' and is_student: # for sd dit, only mask student input.
            visible, invisible, ids_restore, ids_keep = random_masking(x, self.reg_mask_ratio)
            x = visible

        t = self.t_embedder(t)  # (N, hidden_size)
        # only drop labels during training
        y = self.y_embedder(y)  # (N, hidden_size)

        c = t + y  # (N, hidden_size)

        for i,block in enumerate(self.blocks):
            x = block(x, c, rope=self.feat_rope) # [N, num_patches, hidden_size]
            if (i + 1) == depth:
        
                if self.reg_type == 'repa': #linear, direct forward
                    h = self.repa_projector(x) 
                
                elif self.reg_type == 'irepa': # perform spatial normalization
                    H, W = self.input_size // self.patch_size, self.input_size // self.patch_size
                    x = rearrange(x, 'b (h w) d -> b d h w', h=H,w=W)
                    h = self.repa_projector(x)
                    h = rearrange(h, 'b d h w -> b (h w) d')

                    # perform spatial norm
                    gamma = 1.0
                    mean = h.mean(dim=1, keepdim=True)
                    std  = h.std(dim=1, keepdim=True)
                    h = (h - gamma * mean) / (std + 1e-6)

                elif self.reg_type == 'disperse':
                    h = x # (B, N, D)
                elif self.reg_type == 'ig':
                    h = self.final_layer_xr(x, c)
                    h = self.unpatchify(h) # (B, C, H, W)

                elif self.reg_type == 'sra':
                    if is_student: # for stu
                        h = self.final_layer_xr(x)
                    else:
                        h = x # for teacher

                elif self.reg_type == 'sd-dit':
                    h = self.sddit_projector(x) #[B,N*(1-mask_ratio), K]

                    if is_student:
                        all_tokens = torch.cat([x, invisible],dim=1) #[B,N,D]
                        ids_restore = repeat(ids_restore, 'b n -> b n d', d = D)
                        x = torch.gather(all_tokens, dim=1, index=ids_restore)

            # if self.training and self.use_repa and (i + 1) == self.repa_depth:
            #     h = self.repa_projector(x.reshape(-1, D)) 
            #     h = h.reshape(N, T, -1) # [N, num_patches, 768]
        x = self.final_layer(x, c)  # (N, num_patches, patch_size*patch_size*out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        if self.reg_type == 'sd-dit':
            if is_student:
                return x, h, ids_keep
            else:
                return x, h
        else:

            return x, h


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
