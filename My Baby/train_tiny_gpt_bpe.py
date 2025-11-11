# ============================ CONFIG FIRST (before torch import) ============================
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============================ IMPORTS ============================
import json
import math
from pathlib import Path
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
print(f"CUDA Version: {torch.version.cuda}")
print(f"CUDA is available: {torch.cuda.is_available()}")

from tokenizers import Tokenizer
from datasets import load_dataset

from Transformer import gpt as g  # your model

# ======================================== START TRAINING SCRIPT =====================================

# Hyperparameters
block_size       = 256
batch_size       = 8
n_embed          = 256
n_heads          = 4
n_layers         = 6
max_iters        = 1_000_000
eval_interval    = 1000
base_lr          = 1e-4
final_lr         = 1e-10
device           = 'cuda' if torch.cuda.is_available() else 'cpu'

max_loss         = 2.0
patience         = 0
max_patience     = 30000
max_tokens       = 128
grad_accum       = 16

reset_lr = False # to remove later.

# Tokenizer
tokenizer = Tokenizer.from_file("Datasets/bpe_tokenizer.json")
vocab_size = tokenizer.get_vocab_size()

def encode(text: str):
    return tokenizer.encode(text).ids

# ----------------------- Encoded data destinations -----------------------
encoded_data_path = "encoded_data.pt"
SHARDS_DIR = Path("Encoded_Shards")
SHARDS_DIR.mkdir(parents=True, exist_ok=True)
META_PATH = SHARDS_DIR / "meta.json"
CHUNK_TOKENS = 5_000_000

# ============================ ENCODING (chunked) ============================
if not (META_PATH.exists() and any(SHARDS_DIR.glob("encoded_*.pt"))):
    print("🚀 No saved data found. Streaming dataset and encoding into shards...")
    folder_path = "Datasets/"
    try:
        streamed_dataset = load_dataset("text", data_dir=folder_path, split="train", streaming=True)
    except FileNotFoundError:
        print(f"Error: Datasets directory '{folder_path}' not found. Please ensure your dataset is in this folder.")
        sys.exit(1)

buf = []
shard_idx = 0
total_tokens = 0

def flush_shard():
    # Use the `global` keyword to modify the global variables
    global buf, shard_idx, total_tokens
    if not buf:
        return
    t = torch.tensor(buf, dtype=torch.int32)
    shard_path = SHARDS_DIR / f"encoded_{shard_idx:05d}.pt"
    torch.save(t, shard_path)
    print(f"🧱 wrote shard {shard_idx:05d} with {t.numel():,} tokens → {shard_path}")
    total_tokens += t.numel()
    shard_idx += 1
    buf = []

# The rest of your encoding logic remains the same
if os.path.exists(encoded_data_path):
    print("✅ Found saved encoded dataset. Loading single tensor path...")
    train_data = torch.load(encoded_data_path)
    SINGLE_TENSOR_MODE = True
else:
    # If shards already exist from a previous run, we keep them (resume)
    if META_PATH.exists() and any(SHARDS_DIR.glob("encoded_*.pt")):
        print("✅ Found existing shards. Skipping re-encoding.")
    else:
        print("🚀 No saved data found. Streaming dataset and encoding into shards...")
        folder_path = "Datasets/"
        streamed_dataset = load_dataset("text", data_dir=folder_path, split="train", streaming=True)

        for i, example in enumerate(streamed_dataset):
            ids = encode(example["text"])
            buf.extend(ids)

            if i % 1000 == 0:
                print(f"Encoded {i:,} samples… (buffer={len(buf):,} tokens)")

            if len(buf) >= CHUNK_TOKENS:
                flush_shard()

        flush_shard()

        meta = {
            "dtype": "int32",
            "chunk_tokens": CHUNK_TOKENS,
            "tokenizer": "Datasets/bpe_tokenizer.json",
        }
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print("✅ Encoding complete.")

    SINGLE_TENSOR_MODE = False

# ============================ SHARDED DATA ACCESS ============================
class ShardedTokens:
    def __init__(self, shards):
        self.shards = shards
        self._lengths = [int(torch.load(p, map_location="cpu").numel()) for p in self.shards]
        self.cum = [0]
        for L in self._lengths:
            self.cum.append(self.cum[-1] + L)
        self.total_len = self.cum[-1]
        self._cache = {}
        self._cache_order = []
        self._cache_cap = 4

    def __len__(self):
        return self.total_len

    def _locate(self, idx: int):
        lo, hi = 0, len(self.cum) - 2
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.cum[mid] <= idx < self.cum[mid + 1]:
                return mid, idx - self.cum[mid]
            if idx < self.cum[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        raise IndexError("Index out of range")

    def _load_shard(self, shard_id: int):
        if shard_id in self._cache:
            self._cache_order.remove(shard_id)
            self._cache_order.append(shard_id)
            return self._cache[shard_id]
        t = torch.load(self.shards[shard_id], map_location="cpu")
        self._cache[shard_id] = t
        self._cache_order.append(shard_id)
        if len(self._cache_order) > self._cache_cap:
            evict = self._cache_order.pop(0)
            self._cache.pop(evict, None)
        return t

    def get_slice(self, start: int, length: int) -> torch.Tensor:
        remain = length
        pos = start
        out = []
        while remain > 0:
            shard_id, off = self._locate(pos)
            t = self._load_shard(shard_id)
            take = min(remain, self._lengths[shard_id] - off)
            out.append(t[off:off+take])
            pos += take
            remain -= take
        return torch.cat(out, dim=0)

# --- SPLIT DATA INTO TRAIN AND VAL SHARDS ---
all_shards = sorted(SHARDS_DIR.glob("encoded_*.pt"))
assert len(all_shards) > 0, "No encoded shards found."
val_split = 0.05
split_idx = int(len(all_shards) * (1 - val_split))
train_shards = all_shards[:split_idx]
val_shards = all_shards[split_idx:]

TRAIN_DATA = ShardedTokens(train_shards)
VAL_DATA = ShardedTokens(val_shards)

def get_batch(split):
    data = TRAIN_DATA if split == 'train' else VAL_DATA
    x_list, y_list = [], []
    for _ in range(batch_size):
        shard_idx = random.randint(0, len(data.shards) - 1)
        shard_len = data._lengths[shard_idx]
        if shard_len <= block_size + 1:
            continue
        start_in_shard = random.randint(0, shard_len - block_size - 1)
        t = data._load_shard(shard_idx)
        x_piece = t[start_in_shard:start_in_shard+block_size]
        y_piece = t[start_in_shard+1:start_in_shard+block_size+1]
        x_list.append(x_piece)
        y_list.append(y_piece)

    if not x_list: # Handle case where all shards are too small
        return None, None
        
    x = torch.stack(x_list).to(torch.long).to(device)
    y = torch.stack(y_list).to(torch.long).to(device)
    return x, y

# ============================ MODEL / OPT / SCHED ============================
model = g.GPT(vocab_size, block_size, n_embed, n_heads, n_layers).to(device)

best_weights_path = "tinygpt_bpe_best.pt"
checkpoint_path = "checkpoint.pt"

optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

# --- resume if checkpoint exists ---
start_iter = 0
best_val_loss = float("inf")
if os.path.exists(checkpoint_path):
    print("✅ Resuming from checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    best_val_loss = checkpoint.get("best_loss", float("inf"))
    start_iter = checkpoint["iter"] + 1

    if reset_lr: 
        optimizer.param_groups[0]['lr'] = base_lr
        print("Resetting the lr!")
    else:
        print(f"Resuming from iteration {start_iter}, with a best loss of {best_val_loss:.4f}")

elif os.path.exists(best_weights_path):
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    print(f"✅ Loaded best weights from '{best_weights_path}'")

def cosine_lr(step, max_steps, base_lr, final_lr=final_lr):
    if step >= max_steps:
        return final_lr
    decay_ratio = step / max_steps
    cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return final_lr + (base_lr - final_lr) * cosine_decay

@torch.no_grad()
def validate():
    model.eval()
    losses = []
    # Evaluate on a few batches from the validation set
    for _ in range(20): # Number of batches to evaluate
        xb, yb = get_batch('val')
        if xb is None:
            continue
        _, loss = model(xb, yb)
        losses.append(loss.item())
    model.train()
    if losses:
        return sum(losses) / len(losses)
    return float('inf')

# ============================ TRAIN ============================
save_flag = False

print("Starting training...", flush=True)

optimizer.zero_grad()

for iter in range(start_iter, max_iters):
    xb, yb = get_batch('train')
    if xb is None:
        continue

    logits, loss = model(xb, yb)
    
    # [GRAD ACCUM]
    (loss / grad_accum).backward()

    if (iter + 1) % grad_accum == 0:
        lr = cosine_lr(iter, max_iters, base_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        optimizer.zero_grad()

    # --- evaluation and saving ---
    if iter % eval_interval == 0:
        val_loss = validate()
        print(f"Iter {iter}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), best_weights_path)
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter": iter,
                "best_loss": best_val_loss,
            }, checkpoint_path)
            print(f"[Model Saved] New best validation loss: {best_val_loss:.4f}")
        else:
            patience += 1
            print(f"Patience: {patience}/{max_patience}")

        # Your existing sampling code for checking quality
        print("START SAMPLE -------------------------------------")
        # Ensure model is in eval mode for generation
        model.eval()
        eos_token_id = tokenizer.token_to_id("[EOS]")
        prompts = [
            "What is that ",
            "where is this ",
            "what town is in this ",
            "how are you?",
            "what did you do today?",
            "what is your favorite color?",
            "what is your favorite food?",
            "what is your favorite hobby?",
            "do you like iced cream?"
        ]

        for prompt in prompts:
            seed_ids = torch.tensor(encode(prompt)).unsqueeze(0).to(device)
            sample_ids = model.generate(seed_ids, max_new_tokens=max_tokens-32, block_size=block_size, eos_token_id=eos_token_id)
            sample_text = tokenizer.decode(sample_ids[0].tolist())
            print("Sample text:", sample_text)
        
        model.train() # Set back to train mode
        print("END SAMPLE -------------------------------------\n")


    if patience >= max_patience:
        print(f"[Early Stop] Iter {iter}. No improvement for {max_patience} evaluations.")
        break

    if loss.item() <= max_loss:
        print(f"[Early Stop] Iter {iter}. Loss: {loss.item()} Too Low To Continue.")
        break

torch.save(model.state_dict(), "tinygpt_bpe_last.pt")
print("Training done. Final model saved.")