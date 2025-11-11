import torch
import torch.nn as nn
import torch.nn.functional as F
from . import block as b

class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embed, n_heads, n_layers):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        
        self.blocks = nn.Sequential(*[b.Block(n_embed, n_heads, block_size) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.dropout = nn.Dropout(0.1) # add drop out
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)

        x = tok_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        x = self.dropout(x) # add drop out
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, block_size, eos_token_id=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            # Early stopping if EOS is found
            if eos_token_id is not None and (idx_next == eos_token_id).any():
                break

        return idx