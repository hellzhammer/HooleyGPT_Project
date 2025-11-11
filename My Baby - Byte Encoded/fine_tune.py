import torch
import torch.nn as nn
import torch.nn.functional as F
from Transformer import gpt as g
from tokenizers import Tokenizer
import math
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from datasets import load_dataset

# ======================================== START TRAINING SCRIPT =====================================

# Hyperparameters
block_size = 128         
batch_size = 8         
n_embed = 128
n_heads = 4             
n_layers = 4           
max_iters = 50_000       
eval_interval = 200     
learning_rate = 3e-4  # ✅ Lower LR for fine-tuning
final_lr = 5e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

max_loss = 1.0
patience = 0
max_patience = 500
max_tokens = 32

# Tokenizer
tokenizer = Tokenizer.from_file("Datasets/Tokens/bpe_tokenizer.json")
vocab_size = tokenizer.get_vocab_size()

def encode(text):
    return tokenizer.encode(text).ids

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
model = g.GPT(vocab_size, block_size, n_embed, n_heads, n_layers).to(device)

base_weights_path = "tinygpt_bpe_last_fine_tuned.pt" 
if os.path.exists(base_weights_path):
    model.load_state_dict(torch.load(base_weights_path, map_location=device))
    print(f"✅ Loaded base pretrained weights from '{base_weights_path}'")
else:
    print(f"⚠️ Base weights file '{base_weights_path}' not found. Starting from scratch.")

for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        print(f"{name}: bias={module.bias is not None}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Cosine learning rate decay
def cosine_lr(step, max_steps, base_lr, final_lr=final_lr):
    if step >= max_steps:
        return final_lr
    decay_ratio = step / max_steps
    cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return final_lr + (base_lr - final_lr) * cosine_decay

# Manual/best checkpoint saving
save_flag = False
best_loss = float('inf')
fine_tuned_path = "tinygpt_bpe_finetuned_best.pt"

print("Starting fine-tuning...", flush=True)

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
        torch.save(model.state_dict(), fine_tuned_path)
        print(f"[Fine-Tuned Model Saved] Iter {iter}, Loss: {loss.item():.4f}")
    elif save_flag:
        torch.save(model.state_dict(), fine_tuned_path)
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

        eos_token_id = tokenizer.token_to_id("[EOS]")
        prompts = [
            "What is that ",
            "where is this ",
            "what town is in this "
        ]

        for prompt in prompts:
            seed_ids = torch.tensor(encode(prompt)).unsqueeze(0).to(device)
            sample_ids = model.generate(seed_ids, max_new_tokens=max_tokens, block_size=block_size, eos_token_id=eos_token_id)
            sample_text = tokenizer.decode(sample_ids[0].tolist())
            print("Sample text:", sample_text)

        print("END SAMPLE -------------------------------------\n")

torch.save(model.state_dict(), "tinygpt_bpe_last_fine_tuned.pt")
print("Fine-tuning complete. Final model saved.")
