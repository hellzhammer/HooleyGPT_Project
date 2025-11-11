import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, n_embed, ffn_hidden):
        super().__init__()
        self.fc1 = nn.Linear(n_embed, ffn_hidden, bias=False)
        self.fc2 = nn.Linear(ffn_hidden // 2, n_embed, bias=False)

    def forward(self, x):
        x_proj = self.fc1(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        x = F.silu(x1) * x2
        return self.fc2(x)
