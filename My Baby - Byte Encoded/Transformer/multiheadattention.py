import torch
import torch.nn as nn
from . import head as h;

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, block_size, n_embed):
        super().__init__()
        self.heads = nn.ModuleList([h.Head(head_size, block_size, n_embed) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed, bias=False)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)