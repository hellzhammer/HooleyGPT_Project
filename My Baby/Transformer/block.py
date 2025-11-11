import torch.nn as nn
from . import multiheadattention as mh
from . import feedforward as ff

class Block(nn.Module):
    def __init__(self, n_embed, n_heads, block_size):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = mh.MultiHeadAttention(n_heads, head_size, block_size, n_embed)
        self.ffwd = ff.FeedForward(n_embed, ffn_hidden=5504)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x