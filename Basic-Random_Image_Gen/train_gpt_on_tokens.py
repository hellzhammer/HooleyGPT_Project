import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import time

# === hyperparams ===
# Note: block_size = 64*64+1 is extremely large and may cause OOM errors.
# Consider reducing it to a more reasonable size like 256 or 512 for a small model.
block_size = 64*64 + 1
batch_size = 4
max_iters = 1000
eval_interval = 250
learning_rate = 3e-4
# Automatically select device and move tensors to it.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embed = 256
n_heads = 4
n_layers = 4

# === load tokens ===
tokens = np.fromfile("image_tokens.bin", dtype=np.int64)
vocab_size = int(tokens.max()) + 1
data = torch.tensor(tokens, dtype=torch.long)
train_data = data

def get_batch():
    ix = torch.randint(0, len(train_data) - block_size - 1, (batch_size,))
    x = torch.stack([train_data[i:i+block_size] for i in ix])
    y = torch.stack([train_data[i+1:i+block_size+1] for i in ix])
    # Tensors are moved to the device in the training loop now.
    return x, y

# === GPT model (token-level) ===
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        # Use F.scaled_dot_product_attention (fused attention)
        # It's highly optimized and more memory-efficient.
        # It automatically handles the mask and scales the attention weights.
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(), # GELU is a common activation function in modern transformers
            nn.Linear(4 * n_embed, n_embed)
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT_Image(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        self.eval() # Set model to evaluation mode
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
        self.train() # Set model back to training mode
        return idx

model = GPT_Image(vocab_size).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = GradScaler() # Initializes the gradient scaler for mixed precision

for it in range(max_iters):
    xb, yb = get_batch()
    
    # Automatic Mixed Precision (AMP)
    with autocast():
        logits, loss = model(xb.to(device), yb.to(device))
    
    # Gradient scaling for AMP
    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()

    if it % eval_interval == 0:
        print(f"Iter {it} Loss {loss.item():.4f}")

torch.save(model.state_dict(), "gpt_image_small.pt")
print("Saved gpt_image_small.pt")