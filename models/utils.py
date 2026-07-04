import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange, repeat
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




# Actually, no weight decay in JiT.
def add_weight_decay(model, weight_decay=0, skip_list=()):
    decay, no_decay  = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: # frozen param, skip
            continue
        if len(param.shape) == 1 or name.endswith('.bias') or name in skip_list or 'diffloss' in name:
            no_decay.append(param) # no weight decay for bias, norm and diffusion loss
        else:
            decay.append(param)
        
    return [
        {'params':no_decay, 'weight_decay': 0.},
        {'params': decay,   'weight_decay': weight_decay}
    ]


def modulate(x, shift, scale):
    """"
    x: (B, N, D)
    shift, scale: (B,D)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim//2, dtype=np.float64)
    omega /= (embed_dim / 2.0)
    omega = 1. / 10000**omega # (D/2, )
    pos = pos.reshape(-1) # (M, )
    out = np.einsum('m,d->md', pos, omega) # (M, D/2)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)

    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0) # (2, grid_size, grid_size)

    grid = grid.reshape([2, 1, grid_size, grid_size])

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    pos_embed = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        x_fp32 = x.to(torch.float32)
        rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1,keepdim=True) + self.eps)
        res = x_fp32 * rms
        res = (res * self.weight).to(x.dtype)
        return res
        
    
class Attention(nn.Module): 
    # no dropout
    def __init__(self, dim, num_heads, qkv_bias, qk_norm):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias = qkv_bias)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self,x, rope):
        # x: (B, N, D)
        q, k, v = self.qkv(x).chunk(3,dim=-1) # [B,N,D]
        q = rearrange(q, 'B N (heads d) -> B heads N d', heads = self.num_heads)
        k = rearrange(k, 'B N (heads d) -> B heads N d', heads = self.num_heads)
        v = rearrange(v, 'B N (heads d) -> B heads N d', heads = self.num_heads)
        
        q, k =  self.q_norm(q), self.k_norm(k)
        q, k = rope(q), rope(k) #[B, num_heads, N, D//num_heads]

        # attention
        attn_bias = q.new_zeros(q.size(0), 1, q.size(2), k.size(2))
        scale_factor = 1/ math.sqrt(q.size(-1))

        with torch.cuda.amp.autocast(enabled=False):
            attn_weight = q.float() @ k.float().transpose(-2,-1) * scale_factor # [B, num_heads, N, N]
        
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        x = attn_weight @ v # [B, num_heads, N, d]
        x = rearrange(x, 'B num_heads N d -> B N (num_heads d)') # (B, N, D)
        x = self.proj(x)
        return x
    
class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(self, half_dim, pt_seq_len=16, ft_seq_len=None, theta = 10000, num_cls_token = 0):
        super().__init__()
        t = torch.arange(pt_seq_len)
        freqs = 1. / (theta ** (torch.arange(0, half_dim, 2).float() / half_dim)) # 2d
        freqs = t[:,None] * freqs[None,:]
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2) #[seq_len, dim*2]
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim = -1) #[seq_len, seq_len, dim*4]
        
        # simple version for research:
        # freqs = t[:,None] * freqs[None, :]
        # freqs = repeat(freqs, 'h d -> h (d 2)')
        # freqs_h = repeat(freqs, 'h d -> h w d', w = pt_seq_len)
        # freqs_w = repeat(freqs, 'w d -> h w d', h = pt_seq_len)
        # freqs = torch.cat([freqs_h, freqs_w],dim=-1)
        if num_cls_token > 0:
            freqs_flat = freqs.view(-1, freqs.shape[-1])  # [N_img, D]
            cos_img = freqs_flat.cos()
            sin_img = freqs_flat.sin()

            # prepend in-context cls token
            N_img, D = cos_img.shape
            cos_pad = torch.ones(num_cls_token, D, dtype=cos_img.dtype, device=cos_img.device)
            sin_pad = torch.zeros(num_cls_token, D, dtype=sin_img.dtype, device=sin_img.device)
            # give up the positional information, because cos = 1, sin = 0 => forward(x) = x

            self.freqs_cos = torch.cat([cos_pad, cos_img], dim=0).cuda()  # [N_cls+N_img, D]
            self.freqs_sin = torch.cat([sin_pad, sin_img], dim=0).cuda()
        else:
            self.freqs_cos = freqs.cos().view(-1, freqs.shape[-1]).cuda()
            self.freqs_sin = freqs.sin().view(-1, freqs.shape[-1]).cuda()

    def forward(self, x): return  x * self.freqs_cos + rotate_half(x) * self.freqs_sin

