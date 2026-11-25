"""Temporal video tower over per-second DINOv3 features (unchanged from the paper's training code)."""
import math

import torch
from torch import nn
import torch.nn.functional as F


def sinusoid(t, dim):
    freqs = torch.exp(torch.linspace(math.log(2 * math.pi / 2), math.log(2 * math.pi / 2000), dim // 2, device=t.device))
    angle = t[..., None].float() * freqs
    return torch.cat([angle.sin(), angle.cos()], -1)


class VideoTower(nn.Module):
    def __init__(self, width=768, layers=2, heads=8, out=512, use_grid=False, grid_tokens=15, pool='attention'):
        """layers = temporal transformer layers; pool = 'attention' (learned query) or 'mean' (masked mean)."""
        super().__init__()
        self.use_grid, self.width, self.layers, self.pool_mode = use_grid, width, layers, pool
        self.frame = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, width))
        if use_grid:
            self.grid_proj = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, width))
            self.grid_pos = nn.Parameter(torch.zeros(1, grid_tokens, width))
            self.grid_query = nn.Parameter(torch.zeros(1, 1, width))
            self.grid_attention = nn.MultiheadAttention(width, heads, batch_first=True)
            nn.init.trunc_normal_(self.grid_pos, std=.02)
            nn.init.trunc_normal_(self.grid_query, std=.02)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, .1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.query = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.trunc_normal_(self.query, std=.02)
        self.pool = nn.MultiheadAttention(width, heads, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, out))

    def forward(self, g, grid, t, mask):
        """g [B,T,2048], grid [B,T,15,1024] or None, t [B,T] seconds from window start, mask [B,T] True = real second."""
        x = self.frame(g.float())
        if self.use_grid:
            # real seconds only, in chunks (very large attention batches fail inside cuDNN)
            tokens = self.grid_proj(grid[mask].float()) + self.grid_pos
            pooled = [self.grid_attention(self.grid_query.expand(len(c), -1, -1), c, c, need_weights=False)[0][:, 0]
                      for c in tokens.split(16384)]
            x = x.index_put((mask,), x[mask] + torch.cat(pooled).to(x.dtype))
        if self.layers or self.pool_mode == 'attention':
            x = x + sinusoid(t, self.width)
        if self.layers:
            x = self.encoder(x, src_key_padding_mask=~mask)
        if self.pool_mode == 'mean':
            m = mask[..., None].to(x.dtype)
            pooled = (x * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            q = self.query.expand(len(x), -1, -1)
            pooled = self.pool(q, x, x, key_padding_mask=~mask, need_weights=False)[0][:, 0]
        return F.normalize(self.out(pooled).float(), dim=-1)
