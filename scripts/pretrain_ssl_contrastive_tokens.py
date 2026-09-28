import os
import json
import math
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from vqvae_tracks import n_feats, feat_names, sm_labels, data_path, save_dir, TrackVQVAE, prepare_data


vqvae_ckpt = os.path.join(save_dir, "vqvae_best.pt")
scaler_path = os.path.join(save_dir, "scaler.pkl")
out_dir = os.path.join(save_dir, "ssl_tokens")

feat_idx = {name: i for i, name in enumerate(feat_names)}

# Model hyperparameters
latent_dim = 64
model_dim = 64
n_heads = 8
n_layers = 8
ffn_dim = 256
dropout = 0.025
proj_hidden_dim = 64

# Training hyperparameters

batch_size = 512
n_epochs = 15
lr = 1e-3
weight_decay = 1e-6
temperature = 0.5
warmup_steps = 500
grad_clip = 1.0

# augmentation, applied before tokenization
noise_std = {"d0_symlog": 0.05, "z0_symlog": 0.05, "log_pt": 0.03}
track_drop_range = (0.10, 0.20)   # fraction of tracks dropped per view

num_workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def augment_view(tracks, rng):
    n = tracks.shape[0]
    drop_frac = rng.uniform(*track_drop_range)
    keep_n = max(1, int(round(n * (1.0 - drop_frac))))
    keep_idx = np.sort(rng.choice(n, size=keep_n, replace=False))   # to keep pT ordering
    view = tracks[keep_idx].copy()
    for name, std in noise_std.items():
        view[:, feat_idx[name]] += rng.normal(0.0, std, size=view.shape[0]).astype(np.float32)
    return view


class TokenContrastiveDataset(Dataset):
    def __init__(self, feats, offsets, event_ids):
        self.feats, self.offsets, self.event_ids = feats, offsets, event_ids

    def __len__(self):
        return len(self.event_ids)

    def __getitem__(self, i):
        e = self.event_ids[i]
        tracks = self.feats[self.offsets[e]:self.offsets[e + 1]]
        rng = np.random.default_rng()
        return augment_view(tracks, rng), augment_view(tracks, rng)


def worker_init_fn(worker_id):
    base_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(base_seed + worker_id)


def contrastive_collate_fn(batch):
    views_1, views_2 = zip(*batch)
    b = len(views_1)
    max_len = max(max(v.shape[0] for v in views_1), max(v.shape[0] for v in views_2))

    def pad_stack(views):
        padded = np.zeros((b, max_len, n_feats), dtype=np.float32)
        mask = np.zeros((b, max_len), dtype=bool)
        for i, v in enumerate(views):
            padded[i, :v.shape[0]] = v
            mask[i, :v.shape[0]] = True
        return torch.from_numpy(padded), torch.from_numpy(mask)

    x1, mask1 = pad_stack(views_1)
    x2, mask2 = pad_stack(views_2)
    return x1, mask1, x2, mask2


@torch.no_grad()
def tokenize_batch(vqvae, x_raw, mask, mean_t, scale_t):
    x_scaled = (x_raw - mean_t) / scale_t
    z = vqvae.encode(x_scaled, mask)
    _, idx, _, _, _ = vqvae.vq(z, mask)
    return idx.clamp(min=0)


class TokenSetTransformerEncoder(nn.Module):
    def __init__(self, codebook_size, model_dim=model_dim, n_heads=n_heads, n_layers=n_layers,
                 ffn_dim=ffn_dim, latent_dim=latent_dim, dropout=dropout):
        super().__init__()
        self.token_embed = nn.Embedding(codebook_size, model_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads, dim_feedforward=ffn_dim,
                                           dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(model_dim)
        self.out_proj = nn.Linear(model_dim, latent_dim)

    def forward(self, token_ids, mask):
        b = token_ids.shape[0]
        x = self.token_embed(token_ids)
        x = torch.cat([self.cls_token.expand(b, -1, -1), x], dim=1)
        cls_mask = torch.ones(b, 1, dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)
        x = self.transformer(x, src_key_padding_mask=~full_mask)
        return self.out_proj(self.out_norm(x[:, 0, :]))


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=latent_dim, hidden_dim=proj_hidden_dim, out_dim=latent_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, out_dim))

    def forward(self, z):
        return self.net(z)


def simclr_loss(p1, p2, temperature):
    b = p1.shape[0]
    p1 = F.normalize(p1, dim=1)
    p2 = F.normalize(p2, dim=1)
    z = torch.cat([p1, p2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    sim.masked_fill_(torch.eye(2 * b, dtype=torch.bool, device=p1.device), float("-inf"))
    pos_idx = torch.cat([torch.arange(b, 2 * b, device=p1.device), torch.arange(0, b, device=p1.device)])
    return F.cross_entropy(sim, pos_idx)


def train():
    os.makedirs(out_dir, exist_ok=True)
 
    ckpt = torch.load(vqvae_ckpt, map_location=device) # frozen VQVAE
    vqvae = TrackVQVAE(**ckpt["cfg"]).to(device)
    vqvae.load_state_dict(ckpt["state_dict"])
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)
    codebook_size = ckpt["cfg"]["codebook_size"]

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    mean_t = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    scale_t = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)

    feats, offsets, labels = prepare_data(data_path)
    sm_ev = np.where(np.isin(labels, sm_labels))[0]

    train_loader = DataLoader(TokenContrastiveDataset(feats, offsets, sm_ev), batch_size=batch_size, shuffle=True, collate_fn=contrastive_collate_fn, drop_last=True, num_workers=num_workers, worker_init_fn=worker_init_fn, persistent_workers=num_workers > 0)

    encoder = TokenSetTransformerEncoder(codebook_size).to(device)
    proj_head = ProjectionHead().to(device)

    params = list(encoder.parameters()) + list(proj_head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    total_steps = n_epochs * len(train_loader)
    lr_fn = lambda s: min(1.0, (s + 1) / warmup_steps) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(total_steps, 1))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    history = {"epoch": [], "loss": [], "embed_std": []}
    for epoch in range(1, n_epochs + 1):
        encoder.train()
        proj_head.train()
        epoch_loss, n_batches, embed_std = 0.0, 0, None

        for x1_raw, mask1, x2_raw, mask2 in train_loader:
            x1_raw, mask1 = x1_raw.to(device), mask1.to(device)
            x2_raw, mask2 = x2_raw.to(device), mask2.to(device)

            tok1 = tokenize_batch(vqvae, x1_raw, mask1, mean_t, scale_t)
            tok2 = tokenize_batch(vqvae, x2_raw, mask2, mean_t, scale_t)

            z1 = encoder(tok1, mask1)
            z2 = encoder(tok2, mask2)

            if n_batches == 0:
                with torch.no_grad():
                    embed_std = z1.std(dim=0).mean().item()

            loss = simclr_loss(proj_head(z1), proj_head(z2), temperature)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)
        history["embed_std"].append(embed_std)
        print("epoch", epoch, "loss", avg_loss, "embed_std", embed_std, flush=True)

    torch.save(encoder.state_dict(), os.path.join(out_dir, "encoder_ssl_tokens.pt"))
    torch.save(proj_head.state_dict(), os.path.join(out_dir, "proj_head_ssl_tokens.pt"))
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "training_history.csv"), index=False)

if __name__ == "__main__":
    train()
