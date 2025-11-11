import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import sys
import Transformer.gpt as g

# hyper params
block_size = 256
n_embed = 256           
n_heads = 4             
n_layers = 6   

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = Tokenizer.from_file("Datasets/bpe_tokenizer.json")
vocab_size = tokenizer.get_vocab_size()
# script functions
def encode(s): 
    return tokenizer.encode(s).ids
def decode(l): 
    return tokenizer.decode(l)

# INSTANTIATE NEW GPT INSTANCE
model = g.GPT(vocab_size, block_size, n_embed, n_heads, n_layers).to(device)
model.load_state_dict(torch.load("tinygpt_bpe_best.pt", map_location=device))
model.eval()

# GENERATE NEW OUTPUT FROM NEW INPUT
prompt = sys.argv[1] if len(sys.argv) > 1 else "how are you today?"
input_ids = torch.tensor([encode(prompt)], dtype=torch.long).to(device)

# Get EOS token ID from tokenizer
eos_token_id = tokenizer.token_to_id("[EOS]")
# Generate output with early stopping at [EOS]
output_ids = model.generate(input_ids, max_new_tokens=64, block_size=block_size, eos_token_id=eos_token_id)

# Decode and print (cleaning up Ġ)
output_text = decode(output_ids[0].tolist())
print(output_text.replace("Ġ", " ").strip())