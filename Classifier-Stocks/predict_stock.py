
import torch
import torch.nn.functional as F
import sys

block_size = 64
n_embed = 64
n_heads = 4
n_layers = 4
num_classes = 3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Head(torch.nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = torch.nn.Linear(n_embed, head_size, bias=False)
        self.query = torch.nn.Linear(n_embed, head_size, bias=False)
        self.value = torch.nn.Linear(n_embed, head_size, bias=False)
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

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = torch.nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = torch.nn.Linear(n_embed, n_embed)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(torch.nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embed, 4 * n_embed),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * n_embed, n_embed)
        )

    def forward(self, x):
        return self.net(x)

class Block(torch.nn.Module):
    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = torch.nn.LayerNorm(n_embed)
        self.ln2 = torch.nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class EVA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(1000, n_embed)
        self.position_embedding_table = torch.nn.Embedding(block_size, n_embed)
        self.blocks = torch.nn.Sequential(*[Block(n_embed, n_heads) for _ in range(n_layers)])
        self.ln_f = torch.nn.LayerNorm(n_embed)
        self.class_head = torch.nn.Linear(n_embed, num_classes)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.class_head(x[:, -1])
        return logits

# === Load model ===
model = EVA().to(device)
model.load_state_dict(torch.load("stock_model.pt", map_location=device))
model.eval()

# === Input sequence ===
input_str = sys.argv[1] if len(sys.argv) > 1 else "45.12,45.22,45.33,45.40"
features = torch.tensor([[float(x) for x in input_str.split(',')]], dtype=torch.float32)

if features.shape[1] < block_size:
    features = torch.nn.functional.pad(features, (0, block_size - features.shape[1]), value=0)
else:
    features = features[:, :block_size]

input_tensor = features.long().to(device)

with torch.no_grad():
    logits = model(input_tensor)
    predicted_class = torch.argmax(logits, dim=-1).item()

labels = ["SELL", "HOLD", "BUY"]
print("Model says:", labels[predicted_class])
