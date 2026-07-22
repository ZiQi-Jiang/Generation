import numpy as np
import torch
import torch.nn as nn
from typing import Optional

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention



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

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def apply_adaln(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepConditioner(nn.Module):
    def __init__(self, hidden_dim, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[..., None].float() * freqs[None, ...]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        mlp_dtype = next(self.mlp.parameters()).dtype
        if t_freq.dtype != mlp_dtype:
            t_freq = t_freq.to(mlp_dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class ClassEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_dim):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes, hidden_dim)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        x = self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
        return x


def precompute_freqs_cis_2d(dim: int, height: int, width: int, theta: float = 10000.0, scale=16.0):
    x_pos = torch.linspace(0, scale, width)
    y_pos = torch.linspace(0, scale, height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    x_freqs = torch.outer(x_pos, freqs).float()
    y_freqs = torch.outer(y_pos, freqs).float()
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)
    freqs_cis = freqs_cis.reshape(height * width, -1)
    return freqs_cis

def precompute_freqs_cis_ex2d(dim: int, height: int, width:int, theta: float = 10000.0, scale=1.0):
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


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs_cis = freqs_cis[None, :, None, :]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = RMSNorm,
    ) -> None:
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
    def forward(self, x: torch.Tensor, pos, mask) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_emb(q, k, freqs_cis=pos)
        q = q.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()
        v = v.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()

        x = scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
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


class FinalLayer(nn.Module):
    def __init__(self, hidden_dim, out_channels):
        super().__init__()
        self.norm = RMSNorm(hidden_dim, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, out_channels, bias=True)

    @torch.compile
    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x

class PatchTokenEmbedder(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer = None,
            bias: bool = True,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class AugmentedDiTBlock(nn.Module):
    def __init__(self, hidden_dim, groups, mlp_ratio=4.0, adaLN_modulation=None):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim, eps=1e-6)
        self.attn = RotaryAttention(hidden_dim, num_heads=groups, qkv_bias=False)
        self.norm2 = RMSNorm(hidden_dim, eps=1e-6)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = FeedForward(hidden_dim, mlp_hidden_dim)
        self.adaLN_modulation = adaLN_modulation if adaLN_modulation is not None else nn.Sequential(
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )
    @torch.compile
    def forward(self, x, c, pos, mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(apply_adaln(self.norm1(x), shift_msa, scale_msa), pos, mask=mask)
        x = x + gate_mlp * self.mlp(apply_adaln(self.norm2(x), shift_mlp, scale_mlp))
        return x


class PixelTokenEmbedder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim_output: int, use_pixel_abs_pos: bool = True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_dim_output = int(hidden_dim_output)
        self.use_pixel_abs_pos = bool(use_pixel_abs_pos)
        self.proj = nn.Linear(self.in_channels, self.hidden_dim_output, bias=True)
        self._pos_cache = dict()

    def _fetch_pixel_pos_image(self, height: int, width: int, device, dtype):
        if height == width:
            key = ("image", height, width)
            if key in self._pos_cache:
                pe = self._pos_cache[key]
                return pe.to(device=device, dtype=dtype)
            pos_np = get_2d_sincos_pos_embed(self.hidden_dim_output, height)
            pos = torch.from_numpy(pos_np).to(device=device, dtype=dtype)
            self._pos_cache[key] = pos
            return pos
        else:
            key = ("image", height, width)
            if key in self._pos_cache:
                pe = self._pos_cache[key]
                return pe.to(device=device, dtype=dtype)
            grid_h = np.arange(height, dtype=np.float32)
            grid_w = np.arange(width, dtype=np.float32)
            grid = np.meshgrid(grid_w, grid_h)
            grid = np.stack(grid, axis=0).reshape(2, 1, height, width)
            pos_np = get_2d_sincos_pos_embed_from_grid(self.hidden_dim_output, grid)
            pos = torch.from_numpy(pos_np).to(device=device, dtype=dtype)
            self._pos_cache[key] = pos
            return pos

    def forward(self, inputs: torch.Tensor, img_height: int = None, img_width: int = None, patch_size: int = None):
        if inputs.dim() != 4:
            raise ValueError("PixelTokenEmbedder expects inputs of shape [B,C,H,W]")
        assert img_height is not None and img_width is not None and patch_size is not None
        B, C, H, W = inputs.shape
        assert H == img_height and W == img_width
        assert (H % patch_size == 0) and (W % patch_size == 0)
        Hs, Ws = H // patch_size, W // patch_size
        P2 = patch_size * patch_size
        x = inputs.permute(0, 2, 3, 1).contiguous()
        x = self.proj(x)
        if self.use_pixel_abs_pos:
            pos_full = self._fetch_pixel_pos_image(H, W, inputs.device, inputs.dtype)
            pos_full = pos_full.view(H, W, self.hidden_dim_output)
            x = x + pos_full.unsqueeze(0)
        x = x.view(B, Hs, patch_size, Ws, patch_size, self.hidden_dim_output)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B * Hs * Ws, P2, self.hidden_dim_output)
        return x


class PiTBlock(nn.Module):
    def __init__(
        self,
        pixel_hidden_dim: int,
        patch_hidden_dim: int,
        patch_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_hidden_dim: Optional[int] = None,
        attn_num_heads: Optional[int] = None,
        rope_fn=None,
        adaln_post_modulation: bool = False,
    ):
        super().__init__()
        self.pixel_dim = int(pixel_hidden_dim)
        self.context_dim = int(patch_hidden_dim)
        self.patch_size = int(patch_size)
        self.attn_dim = int(attn_hidden_dim) if attn_hidden_dim is not None else self.context_dim
        self.num_heads = int(attn_num_heads) if attn_num_heads is not None else int(num_heads)
        assert (
            self.attn_dim % self.num_heads == 0
        ), "pixel attention hidden size must be divisible by pixel num_heads"
        p2 = self.patch_size * self.patch_size
        self.compress_to_attn = nn.Linear(p2 * self.pixel_dim, self.attn_dim, bias=True)
        self.expand_from_attn = nn.Linear(self.attn_dim, p2 * self.pixel_dim, bias=True)
        self.norm1 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.attn = RotaryAttention(self.attn_dim, num_heads=self.num_heads, qkv_bias=False)
        self.norm2 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.mlp = MLP(self.pixel_dim, mlp_ratio=mlp_ratio, drop=0.0)
        self.adaln_post_modulation = bool(adaln_post_modulation)
        n_mod = 4 if self.adaln_post_modulation else 6
        self.adaLN_modulation = nn.Sequential(nn.Linear(self.context_dim, n_mod * self.pixel_dim * p2, bias=True))
        self._pos_cache = dict()
        self._rope_fn = rope_fn if rope_fn is not None else precompute_freqs_cis_2d

    def _fetch_pos(self, height: int, width: int, device):
        key = (height, width)
        if key in self._pos_cache:
            return self._pos_cache[key].to(device)
        pos = self._rope_fn(self.attn_dim // self.num_heads, height, width).to(device)
        self._pos_cache[key] = pos
        return pos
    @torch.compile
    def forward(self, x: torch.Tensor, s_cond: torch.Tensor, image_height: int, image_width: int, patch_size: int, mask=None) -> torch.Tensor:
        BL, P2, C = x.shape
        if C != self.pixel_dim:
            raise ValueError(f"PiTBlock expected pixel_dim={self.pixel_dim}, got {C}")
        assert (image_height % patch_size == 0) and (image_width % patch_size == 0)
        Hs, Ws = image_height // patch_size, image_width // patch_size
        L = Hs * Ws
        B = BL // L
        n_mod = 4 if self.adaln_post_modulation else 6
        cond_params = self.adaLN_modulation(s_cond).view(BL, P2, n_mod * self.pixel_dim)
        if self.adaln_post_modulation:
            scale1, shift1, scale2, shift2 = torch.chunk(cond_params, 4, dim=-1)
            x_norm = self.norm1(x)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(cond_params, 6, dim=-1)
            x_norm = apply_adaln(self.norm1(x), shift_msa, scale_msa)
        x_flat = x_norm.view(BL, P2 * self.pixel_dim)
        x_comp = self.compress_to_attn(x_flat).view(B, L, self.attn_dim)
        pos_comp = self._fetch_pos(Hs, Ws, x.device)
        attn_out = self.attn(x_comp, pos_comp, mask)
        attn_flat = self.expand_from_attn(attn_out.view(B * L, self.attn_dim))
        attn_exp = attn_flat.view(BL, P2, self.pixel_dim)
        if self.adaln_post_modulation:
            x = x + attn_exp * (1 + scale1) + shift1
            mlp_out = self.mlp(self.norm2(x))
            x = x + mlp_out * (1 + scale2) + shift2
        else:
            x = x + gate_msa * attn_exp
            mlp_out = self.mlp(apply_adaln(self.norm2(x), shift_mlp, scale_mlp))
            x = x + gate_mlp * mlp_out
        return x


class PixDiT(nn.Module):
    def __init__(
        self,
        input_size=256,
        mlp_ratio=4.0,
        in_channels=4,
        num_heads=12,
        hidden_dim=1152,
        pixel_hidden_dim=64,
        patch_depth=18,
        pixel_depth=4,
        patch_size=2,
        num_classes=1000,
        use_pixel_abs_pos=True,
        pit_adaln_post_modulation=False,
        reg_type=None,
        reg_depth=8,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.patch_depth = int(patch_depth)
        self.pixel_depth = int(pixel_depth)
        self.patch_size = int(patch_size)
        self.pixel_hidden_dim = int(pixel_hidden_dim)
        self.num_classes = int(num_classes)
        self.use_pixel_abs_pos = bool(use_pixel_abs_pos)
        self.pit_adaln_post_modulation = bool(pit_adaln_post_modulation)
        self.reg_type = str(reg_type)
        self.reg_depth = int(reg_depth)
        if self.pixel_depth <= 0:
            raise ValueError("PixDiT expects pixel_depth > 0 to preserve the dual-level pipeline")

        self.pixel_embedder = PixelTokenEmbedder(self.in_channels, self.pixel_hidden_dim, use_pixel_abs_pos=self.use_pixel_abs_pos)
        self.s_embedder = PatchTokenEmbedder(self.in_channels * self.patch_size ** 2, self.hidden_dim, bias=True)
        self.t_embedder = TimestepConditioner(self.hidden_dim)
        self.y_embedder = ClassEmbedder(self.num_classes + 1, self.hidden_dim)

        self.final_layer = FinalLayer(self.pixel_hidden_dim, self.out_channels)
        self.patch_blocks = nn.ModuleList(
            [AugmentedDiTBlock(self.hidden_dim, self.num_heads) for _ in range(self.patch_depth)]
        )
        self.pixel_blocks = nn.ModuleList(
            [
                PiTBlock(
                    self.pixel_hidden_dim,
                    self.hidden_dim,
                    patch_size=self.patch_size,
                    num_heads=self.num_heads,
                    mlp_ratio=4.0,
                    adaln_post_modulation=self.pit_adaln_post_modulation,
                )
                for _ in range(self.pixel_depth)
            ]
        )
        if self.reg_type == 'repa':
            # self.repa_depth = 8 if  depth >=24 else 4
            self.repa_projector = nn.Sequential(
            nn.Linear(self.hidden_dim, 2048), nn.SiLU(),
            nn.Linear(2048, 2048), nn.SiLU(),
            nn.Linear(2048, 768),
            )

            self.repa_projector.compile()
            
        self.initialize_weights()
        self.precompute_pos = dict()

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        else:
            pos = precompute_freqs_cis_2d(self.hidden_dim // self.num_heads, height, width).to(device)
            self.precompute_pos[(height, width)] = pos
            return pos

    def initialize_weights(self):
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)
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

    def forward(self, x, t, y, s=None, mask=None):
        B, _, H, W = x.shape
        pos = self.fetch_pos(H // self.patch_size, W // self.patch_size, x.device)
        x_patches = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        t_emb = self.t_embedder(t.view(-1)).view(B, -1, self.hidden_dim)
        y_emb = self.y_embedder(y).view(B, 1, self.hidden_dim)
        c = nn.functional.silu(t_emb + y_emb)
        if s is None:
            s = self.s_embedder(x_patches)
            for i, block in enumerate(self.patch_blocks):
                s = block(s, c, pos, mask)
                   # repa
                if self.training and self.reg_type!="none" and (i + 1) == self.reg_depth:
                    if self.reg_type == "repa": #linear, direct forward
                        h = self.repa_projector(s) 
                    else:
                        raise NotImplementedError()

            s = nn.functional.silu(t_emb + s)
        batch_size, length, _ = s.shape
        s_cond = s.view(batch_size * length, self.hidden_dim)
        x_pixels = self.pixel_embedder(x, img_height=H, img_width=W, patch_size=self.patch_size)
        for blk in self.pixel_blocks:
            x_pixels = blk(x_pixels, s_cond, H, W, self.patch_size, mask)
        x_pixels = self.final_layer(x_pixels)
        C_out = self.out_channels
        P2 = self.patch_size * self.patch_size
        x_pixels = x_pixels.view(B, length, P2, C_out).permute(0, 3, 2, 1).contiguous()
        x_pixels = x_pixels.view(B, C_out * P2, length)
        x_img = torch.nn.functional.fold(x_pixels, (H, W), kernel_size=self.patch_size, stride=self.patch_size)
        
        if self.reg_type != "none"  and self.training:
            return x_img, h
        else:
            return x_img
    



def PiT_B_16(**kwargs):
    return PixDiT(patch_depth=12, pixel_depth=2, hidden_dim=768, num_heads=12, pixel_hidden_dim=16, patch_size=16, **kwargs)

def PiT_B_4(**kwargs):
    return PixDiT(patch_depth=12, pixel_depth=2, hidden_dim=768, num_heads=12, pixel_hidden_dim=16, patch_size=4,  **kwargs)

def PiT_B_32(**kwargs):
    return PixDiT(patch_depth=12, pixel_depth=2, hidden_dim=768, num_heads=12, pixel_hidden_dim=16, patch_size=32,  **kwargs)

def PiT_L_16(**kwargs):
    return PixDiT(patch_depth=22, pixel_depth=4, hidden_dim=1024, num_heads=16, pixel_hidden_dim=16, patch_size=16,  **kwargs)

def PiT_XL_16(**kwargs):
    return PixDiT(patch_depth=26, pixel_depth=4, hidden_dim=1152, num_heads=16, pixel_hidden_dim=16, patch_size=16,  **kwargs)



PiT_models = {
    "PiT-B/16": PiT_B_16,
    "PiT-B/4": PiT_B_4,
    "PiT-B/32": PiT_B_32,
    "PiT-L/16": PiT_L_16,
    "PiT-XL/16": PiT_XL_16
}