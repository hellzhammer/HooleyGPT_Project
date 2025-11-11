# vqvae.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_ch=3, hidden=128, z_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden//2, 4, 2, 1), # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(hidden//2, hidden, 4, 2, 1), # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(hidden, z_dim, 3, 1, 1), # keep 8x8
        )
    def forward(self, x): return self.net(x)  # B x z_dim x 8 x 8

class Decoder(nn.Module):
    def __init__(self, out_ch=3, hidden=128, z_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(z_dim, hidden, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden, hidden//2, 4, 2, 1), # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(hidden//2, out_ch, 4, 2, 1), # 16 -> 32
            nn.Sigmoid()  # images normalized 0..1
        )
    def forward(self, z): return self.net(z)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, z):  # z: B x D x H x W
        B, D, H, W = z.shape
        z_flat = z.permute(0,2,3,1).contiguous().view(-1, D)  # (B*H*W) x D

        # compute distances
        embed = self.embedding.weight  # K x D
        dists = (z_flat.pow(2).sum(1, keepdim=True)
                - 2*z_flat @ embed.t()
                + embed.pow(2).sum(1).unsqueeze(0))  # (BHW) x K

        indices = torch.argmin(dists, dim=1)  # (BHW)
        z_q = embed[indices].view(B, H, W, D).permute(0,3,1,2).contiguous()  # B x D x H x W

        # loss
        z_flat_detach = z_flat.detach()
        embed_detach = embed.detach()
        commitment_loss = F.mse_loss(z, z_q.detach())
        embed_loss = F.mse_loss(z_flat_detach, embed[indices])

        loss = embed_loss + self.beta * commitment_loss

        # straight-through estimator
        z_q_st = z + (z_q - z).detach()

        # return quantized, loss, and indices reshaped to B x H x W (integers)
        codes = indices.view(B, H, W)
        return z_q_st, loss, codes

class VQVAE(nn.Module):
    def __init__(self, in_ch=3, hidden=128, z_dim=64, num_embeddings=512, beta=0.25):
        super().__init__()
        self.encoder = Encoder(in_ch, hidden, z_dim)
        self.quant = VectorQuantizer(num_embeddings, z_dim, beta)
        self.decoder = Decoder(in_ch, hidden, z_dim)

    def forward(self, x):
        z = self.encoder(x)                      # B x D x H x W (H=W=8 for 32x32)
        z_q, vq_loss, codes = self.quant(z)     # codes: B x H x W (integers 0..K-1)
        x_rec = self.decoder(z_q)               # B x 3 x 32 x 32
        recon_loss = F.mse_loss(x_rec, x)
        loss = recon_loss + vq_loss
        return x_rec, loss, codes
