
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

print("Starting stock predictor training...")

# === Hyperparameters ===
block_size = 64
batch_size = 8
max_iters = 2000
eval_interval = 200
learning_rate = 0.0001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embed = 64
n_heads = 4
n_layers = 4
num_classes = 3  # SELL, HOLD, BUY

# === Load CSV dataset ===
df = pd.read_csv("stock_data_100.csv")
label_map = {"SELL": 0, "HOLD": 1, "BUY": 2}

def encode_features(row):
    return torch.tensor([float(x) for x in row["features"].split(',')], dtype=torch.float32)

data_x = torch.stack([encode_features(r) for _, r in df.iterrows()])
data_y = torch.tensor([label_map[r["label"]] for _, r in df.iterrows()], dtype=torch.long)

def pad(x):
    if len(x) >= block_size:
        return x[:block_size]
    else:
        return F.pad(x, (0, block_size - len(x)), value=0)

data_x = torch.stack([pad(x) for x in data_x])

def get_batch():
    ix = torch.randint(len(data_x), (batch_size,))
    x = torch.stack([data_x[i] for i in ix]).to(device)
    y = torch.stack([data_y[i] for i in ix]).to(device)
    return x.long(), y

# === Transformer Model ===
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) / (C ** 0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v

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
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed)
        )
    def forward(self, x):
        return self.net(x)

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

class EVA(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(1000, n_embed)  # input float tokens approximated
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.class_head = nn.Linear(n_embed, num_classes)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.class_head(x[:, -1])
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits, targets)
        return logits, loss

model = EVA().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if iter % eval_interval == 0:
        print(f"Iter {iter}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "stock_model.pt")
print("Training complete.")
