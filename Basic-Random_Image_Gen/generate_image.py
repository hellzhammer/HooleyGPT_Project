# generate_image.py
import torch, numpy as np
from vqvae import VQVAE
from train_gpt_on_tokens import GPT_Image, block_size  # reuse block_size
from torchvision.utils import save_image
device = 'cuda' if torch.cuda.is_available() else 'cpu'

K = 512
EOS = K
vocab_size = K + 1

# load models
vq = VQVAE(in_ch=3, hidden=128, z_dim=64, num_embeddings=K).to(device)
vq.load_state_dict(torch.load("vqvae_small.pt")["model_state"])
vq.eval()

gpt = GPT_Image(vocab_size).to(device)
gpt.load_state_dict(torch.load("gpt_image_small.pt"))
gpt.eval()

# create a start token (empty or random); here we'll seed with EOS and then sample
start = torch.tensor([[EOS]* (block_size//2)], dtype=torch.long).to(device)  # seed
out = gpt.generate(start, max_new_tokens=block_size)  # produces start + new tokens
tokens = out[0, -block_size:].cpu().numpy().astype(np.int64)  # length = block_size

# remove trailing EOS if exists
tokens = [int(t) for t in tokens if t != EOS][: (32*32)]  # keep H*W tokens

# turn tokens into codes H x W
codes = np.array(tokens, dtype=np.int64)
if codes.size < 32*32:
    # pad with zeros
    codes = np.pad(codes, (0, 32*32 - codes.size), 'constant', constant_values=0)
codes = codes.reshape(1, 32, 32)

codes_t = torch.tensor(codes, dtype=torch.long).to(device)
# lookup embedding vectors for quantized codes
embedding = vq.quant.embedding.weight  # K x D
z_q = embedding[codes_t].permute(0,3,1,2).contiguous().float()  # B x D x H x W

with torch.no_grad():
    img = vq.decoder(z_q)  # B x 3 x 32 x 32
    save_image(img, "generated.png")
print("Saved generated.png")
