import torch
import torch.nn.functional as F
import sys
import Transformer.gpt as g

# ====================== START MAIN SCRIPT ==============================================

# Hyperparameters
block_size = 128
n_embed = 128           
n_heads = 8             
n_layers = 4   

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import unicodedata
import re

def normalize_text(text: str) -> str:
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Optional: Lowercase
    text = text.lower()

    # Optional: Remove control characters (except newline/tab)
    text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)

    return text

def encode(text: str) -> list:
    normalized = normalize_text(text)
    return list(normalized.encode("utf-8"))

def decode(bytes_list: list) -> str:
    return bytes(bytes_list).decode("utf-8", errors="ignore")


vocab_size = 256  # Byte-level vocabulary

# Instantiate model
model = g.GPT(vocab_size, block_size, n_embed, n_heads, n_layers).to(device)
model.load_state_dict(torch.load("tinygpt_bpe_best.pt", map_location=device))
model.eval()

# Generate new output from input
prompt = sys.argv[1] if len(sys.argv) > 1 else "[BOS] what time is it [SEP]"
input_ids = torch.tensor([encode(prompt)], dtype=torch.long).to(device)

# EOS token ID (optional): you can define one or just omit it
eos_token_id = ord("~")  # example byte for [EOS], or set to None

# Generate output
output_ids = model.generate(input_ids, max_new_tokens=200, block_size=block_size, eos_token_id=eos_token_id)

# Decode and print output
output_text = decode(output_ids[0].tolist())
print(output_text.strip())
