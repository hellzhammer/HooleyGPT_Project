# train_vqvae.py
import torch, os, time
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from vqvae import VQVAE

# --- Definitions can stay outside the main block ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
epochs = 20
lr = 2e-4
image_size = 256

data_dir = "images"

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),  # 0..1
])

# --- Main execution logic must be inside this block ---
if __name__ == '__main__':
    print("Looking for images in:", os.path.abspath(data_dir))
    print("Exists?", os.path.exists(data_dir))
    print("Is folder?", os.path.isdir(data_dir))

    # Make sure to have the correct folder structure (images/data/...)
    ds = datasets.ImageFolder(os.path.abspath(data_dir), transform=transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = VQVAE(in_ch=3, hidden=128, z_dim=64, num_embeddings=512, beta=0.25).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running = 0.0
        t0 = time.time()
        for i, (x, _) in enumerate(dl):
            x = x.to(device)
            x_rec, loss, codes = model(x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            if (i+1) % 100 == 0:
                print(f"Epoch {epoch} batch {i+1} loss {running / 100:.4f}")
                running = 0.0
        print(f"Epoch {epoch} finished in {time.time()-t0:.1f}s")

    torch.save({
        "model_state": model.state_dict()
    }, "vqvae_small.pt")
    print("Saved vqvae_small.pt")