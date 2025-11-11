import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Head(nn.Module):
    def __init__(self, head_size, block_size, n_embed):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.head_size = head_size

    def apply_rotary_embedding(self, x):
        B, T, C = x.shape
        # Prepare sinusoidal frequencies
        theta = 10000 ** (torch.arange(0, C, 2, device=x.device) / C)
        pos = torch.arange(T, device=x.device).unsqueeze(1)  # [T, 1]
        freq = pos / theta  # [T, C/2]
        sin = torch.sin(freq)
        cos = torch.cos(freq)

        # Reshape for broadcasting
        sin = sin.unsqueeze(0).repeat(B, 1, 1)  # [B, T, C/2]
        cos = cos.unsqueeze(0).repeat(B, 1, 1)

        x1, x2 = x[..., ::2], x[..., 1::2]  # Split into even/odd
        x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rot

    def forward(self, x):
        B, T, C = x.shape
        k = self.apply_rotary_embedding(self.key(x))
        q = self.apply_rotary_embedding(self.query(x))
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        return wei @ v