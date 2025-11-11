import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from vqvae import VQVAE
import numpy as np

# --- Definitions can stay outside the main block ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_size = 256
data_dir = "images"
batch_size = 64
K = 512  # must match your vqvae num_embeddings
EOS = K  # use K as EOS token

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# --- Main execution logic must be inside this block ---
if __name__ == '__main__':
    ds = datasets.ImageFolder(data_dir, transform=transform)
    # Using pin_memory=True is often a good practice for speed with CUDA
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = VQVAE(in_ch=3, hidden=128, z_dim=64, num_embeddings=K, beta=0.25).to(device)
    ckpt = torch.load("vqvae_small.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_tokens = []
    with torch.no_grad():
        for x, _ in dl:
            x = x.to(device)
            z = model.encoder(x)
            _, _, codes = model.quant(z)
            # codes: B x H x W
            B, H, W = codes.shape
            codes = codes.view(B, -1).cpu().numpy().astype(np.int64)  # flatten H*W
            for i in range(B):
                # append tokens for that image then EOS
                all_tokens.extend(codes[i].tolist())
                all_tokens.append(EOS)

    # save as a binary int64 file
    np.array(all_tokens, dtype=np.int64).tofile("image_tokens.bin")
    print("Saved image_tokens.bin, length:", len(all_tokens))