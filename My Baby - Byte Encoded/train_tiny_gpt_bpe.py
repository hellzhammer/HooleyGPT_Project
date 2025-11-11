import torch
import torch.nn as nn
import torch.nn.functional as F
from Transformer import gpt as g
#from tokenizers import Tokenizer
import math
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from datasets import load_dataset

# ======================================== START TRAINING SCRIPT =====================================

# Hyperparameters
block_size = 128         
batch_size = 8         
n_embed = 128
n_heads = 8             
n_layers = 4           
max_iters = 50_000       
eval_interval = 200     
learning_rate = 4e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

max_loss = 0.1
patience = 0
max_patience = 250
max_tokens = 64

import unicodedata
import re

def normalize_text(text: str) -> str:
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Optional: Remove control characters (except newline/tab)
    text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)

    return text

def encode(text: str) -> list:
    normalized = normalize_text(text)
    return list(normalized.encode("utf-8"))

def decode(bytes_list: list) -> str:
    return bytes(bytes_list).decode("utf-8", errors="ignore")


vocab_size = 256  # One value for each byte

# Load or encode dataset
encoded_data_path = "encoded_data.pt"

if os.path.exists(encoded_data_path):
    print("✅ Found saved encoded dataset. Loading...")
    train_data = torch.load(encoded_data_path)
else:
    print("🚀 No saved data found. Streaming dataset and encoding...")

    # Load dataset in streaming mode
    folder_path = "Datasets/"
    streamed_dataset = load_dataset("text", data_dir=folder_path, split="train", streaming=True)

    # Encode entire dataset incrementally
    data_ids = []
    for i, example in enumerate(streamed_dataset):
        ids = encode(example["text"])
        data_ids.extend(ids)

        if i % 1000 == 0:
            print(f"Encoded {i} samples... (Current length: {len(data_ids)})")

    # Convert to tensor and save
    train_data = torch.tensor(data_ids, dtype=torch.long)
    torch.save(train_data, encoded_data_path)
    print(f"✅ Encoding complete. Saved to '{encoded_data_path}'")

def get_batch():
    ix = torch.randint(len(train_data) - block_size, (batch_size,))
    x = torch.stack([train_data[i:i+block_size] for i in ix])
    y = torch.stack([train_data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# Build model
model = g.GPT(vocab_size=256, block_size=block_size, n_embed=n_embed, n_heads=n_heads, n_layers=n_layers).to(device)

# Load best weights if available
best_weights_path = "tinygpt_bpe_best.pt"
if os.path.exists(best_weights_path):
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    print(f"✅ Loaded best weights from '{best_weights_path}'")

for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        print(f"{name}: bias={module.bias is not None}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Cosine learning rate decay
def cosine_lr(step, max_steps, base_lr, final_lr=1e-5):
    if step >= max_steps:
        return final_lr
    decay_ratio = step / max_steps
    cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return final_lr + (base_lr - final_lr) * cosine_decay

# Spacebar/manual save + best checkpoint
save_flag = False
best_loss = float('inf')

print("Starting training...", flush=True)

for iter in range(max_iters):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()

    lr = cosine_lr(iter, max_iters, learning_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()

    if loss.item() < best_loss:
        best_loss = loss.item()
        patience = 0
        torch.save(model.state_dict(), best_weights_path)
        print(f"[Model Saved] Iter {iter}, Loss: {loss.item():.4f}")
    elif save_flag:
        torch.save(model.state_dict(), best_weights_path)
        print(f"[Manual Save] Iter {iter}, Loss: {loss.item():.4f}")
        save_flag = False
    else:
        patience += 1

    if patience >= max_patience:
        print(f"[Early Stop] Iter {iter}. No improvement for {max_patience} evaluations.")
        break

    if loss.item() <= max_loss:
        print(f"[Early Stop] Iter {iter}. Loss: {loss.item()} Too Low To Continue.")
        break

    if iter % eval_interval == 0:
        print(f"Iter {iter}, Loss: {loss.item():.4f}, LR: {lr:.6f}")
        print("START SAMPLE -------------------------------------")

        prompts = [
            "What is that ",
            "where is this ",
            "what town is in this "
        ]

        for prompt in prompts:
            seed_ids = torch.tensor(encode(prompt)).unsqueeze(0).to(device)
            sample_ids = model.generate(seed_ids, max_new_tokens=max_tokens, block_size=block_size)
            sample_text = decode(sample_ids[0].tolist())

            print("Sample text:", sample_text)

        print("END SAMPLE -------------------------------------\n")

torch.save(model.state_dict(), "tinygpt_bpe_last.pt")
print("Training done. Final model saved.")
