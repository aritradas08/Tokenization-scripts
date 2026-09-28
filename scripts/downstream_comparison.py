import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vqvae_tracks import TrackVQVAE, sm_labels, max_tracks, num_workers, device, data_path, save_dir
from pretrain_ssl_contrastive import SetTransformerEncoder as ContinuousEncoder, latent_dim
from pretrain_ssl_contrastive_tokens import TokenSetTransformerEncoder

#paths
cont_dir = os.path.join(save_dir, "checkpoints")
cont_ckpt = os.path.join(cont_dir, "encoder_ssl_simclr.pt")
cont_scaler_path = os.path.join(cont_dir, "scaler.pkl")
vqvae_ckpt = os.path.join(save_dir, "vqvae_best.pt")
vqvae_scaler_path = os.path.join(save_dir, "scaler.pkl")
tok_ckpt = os.path.join(save_dir, "ssl_tokens", "encoder_ssl_tokens.pt")
tokens_npz = os.path.join(save_dir, "tokens_all_events.npz")
out_dir = os.path.join(save_dir, "downstream_compare")

raw_cols = ["d0", "z0", "theta", "p", "eta", "phi", "pt"]
extract_batch = 512

# probe settings
n_seeds = 5
probe_epochs = 40
probe_batch_size = 256
train_sizes = [100, 1000, 10000, 100000]


def symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def load_sorted_truncated_events(path):
    df = pd.read_parquet(path)
    lengths = np.array([len(a) for a in df[raw_cols[0]].values], dtype=np.int64)
    keep_ev = lengths > 0
    df, lengths = df[keep_ev], lengths[keep_ev]
    labels = df["label"].values.astype(np.int64)
    cols = {c: np.concatenate(df[c].values).astype(np.float64) for c in raw_cols}
    del df

    n_events = len(lengths)
    ev = np.repeat(np.arange(n_events), lengths)
    order = np.lexsort((-cols["pt"], ev))
    cols = {k: v[order] for k, v in cols.items()}

    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    rank = np.arange(len(ev)) - np.repeat(starts, lengths)
    cols = {k: v[rank < max_tracks] for k, v in cols.items()}

    new_len = np.minimum(lengths, max_tracks)
    offsets = np.concatenate([[0], np.cumsum(new_len)]).astype(np.int64)
    return cols, offsets, labels


def engineer_continuous(cols):
    phi, theta = cols["phi"], cols["theta"]
    tx = np.cos(phi) * np.sin(theta)
    ty = np.sin(phi) * np.sin(theta)
    tz = np.cos(theta)
    return np.column_stack([symlog(cols["d0"]), symlog(cols["z0"]), cols["p"], cols["pt"], tx, ty, tz]).astype(np.float32)


def engineer_tokens(cols, offsets):
    lengths = np.diff(offsets)
    phi_ref = np.repeat(cols["phi"][offsets[:-1]], lengths)
    dphi = (cols["phi"] - phi_ref + np.pi) % (2 * np.pi) - np.pi
    return np.column_stack([
        symlog(cols["d0"]),
        symlog(cols["z0"]),
        np.log(np.clip(cols["pt"], 1e-3, None)),
        cols["eta"],
        np.cos(dphi),
        np.sin(dphi),
    ]).astype(np.float32)


class EventSetDataset(Dataset):
    def __init__(self, feats, offsets, event_ids):
        self.feats, self.offsets, self.event_ids = feats, offsets, event_ids

    def __len__(self):
        return len(self.event_ids)

    def __getitem__(self, i):
        e = self.event_ids[i]
        return self.feats[self.offsets[e]:self.offsets[e + 1]], e


def collate(batch):
    xs, evs = zip(*batch)
    b, fdim = len(xs), xs[0].shape[1]
    longest = max(x.shape[0] for x in xs)
    x = np.zeros((b, longest, fdim), dtype=np.float32)
    m = np.zeros((b, longest), dtype=bool)
    for i, a in enumerate(xs):
        x[i, :len(a)] = a
        m[i, :len(a)] = True
    return torch.from_numpy(x), torch.from_numpy(m), torch.as_tensor(np.array(evs), dtype=torch.long)


@torch.no_grad()
def extract_continuous_embeddings(feats, offsets, n_events):
    with open(cont_scaler_path, "rb") as f:
        scaler = pickle.load(f)
    feats_scaled = scaler.transform(feats).astype(np.float32)

    encoder = ContinuousEncoder().to(device)
    encoder.load_state_dict(torch.load(cont_ckpt, map_location=device))
    encoder.eval()

    z_out = np.zeros((n_events, latent_dim), dtype=np.float32)
    loader = DataLoader(EventSetDataset(feats_scaled, offsets, np.arange(n_events)), batch_size=extract_batch, shuffle=False, collate_fn=collate, num_workers=num_workers)
    for x, m, ev in loader:
        x, m = x.to(device), m.to(device)
        z_out[ev.numpy()] = encoder(x, m).cpu().numpy()
    return z_out


@torch.no_grad()
def extract_token_embeddings(feats, offsets, n_events):
    with open(vqvae_scaler_path, "rb") as f:
        vq_scaler = pickle.load(f)
    mean_t = torch.tensor(vq_scaler.mean_, dtype=torch.float32, device=device)
    scale_t = torch.tensor(vq_scaler.scale_, dtype=torch.float32, device=device)

    vq_ckpt = torch.load(vqvae_ckpt, map_location=device)
    vqvae = TrackVQVAE(**vq_ckpt["cfg"]).to(device)
    vqvae.load_state_dict(vq_ckpt["state_dict"])
    vqvae.eval()

    tok_encoder = TokenSetTransformerEncoder(vq_ckpt["cfg"]["codebook_size"]).to(device)
    tok_encoder.load_state_dict(torch.load(tok_ckpt, map_location=device))
    tok_encoder.eval()

    z_out = np.zeros((n_events, latent_dim), dtype=np.float32)
    loader = DataLoader(EventSetDataset(feats, offsets, np.arange(n_events)), batch_size=extract_batch,
                        shuffle=False, collate_fn=collate, num_workers=num_workers)
    for x, m, ev in loader:
        x, m = x.to(device), m.to(device)
        z = vqvae.encode((x - mean_t) / scale_t, m)
        _, idx, _, _, _ = vqvae.vq(z, m)
        z_out[ev.numpy()] = tok_encoder(idx.clamp(min=0), m).cpu().numpy()
    return z_out


def run_extract():
    os.makedirs(out_dir, exist_ok=True)
    cols, offsets, labels = load_sorted_truncated_events(data_path)
    n_events = len(labels)

    z_cont = extract_continuous_embeddings(engineer_continuous(cols), offsets, n_events)
    z_tok = extract_token_embeddings(engineer_tokens(cols, offsets), offsets, n_events)

    z_vq = np.load(tokens_npz)["z_mean"]

    np.savez(os.path.join(out_dir, "embeddings.npz"), z_continuous=z_cont, z_token=z_tok, z_vqvae_encoder=z_vq, labels=labels, sm_labels=np.array(sm_labels))
    return {"continuous": z_cont, "token": z_tok, "vqvae_encoder": z_vq}, labels


class MLPProbe(nn.Module):
    def __init__(self, in_dim, n_classes, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes))

    def forward(self, x):
        return self.net(x)


def make_val_test_split(sm_event_ids, val_frac=0.15, test_frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    ev = sm_event_ids.copy()
    rng.shuffle(ev)
    n = len(ev)
    n_val, n_test = int(val_frac * n), int(test_frac * n)
    val_ev = np.sort(ev[:n_val])
    test_ev = np.sort(ev[n_val:n_val + n_test])
    train_pool = np.sort(ev[n_val + n_test:])
    return train_pool, val_ev, test_ev


def train_probe(z, labels, label_to_idx, train_ev, val_ev, test_ev, n_classes, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    def make_xy(event_ids):
        x = z[event_ids]
        y = np.array([label_to_idx[labels[e]] for e in event_ids])
        return torch.from_numpy(x).float(), torch.from_numpy(y).long()

    x_train, y_train = make_xy(train_ev)
    x_val, y_val = make_xy(val_ev)
    x_test, y_test = make_xy(test_ev)

    model = MLPProbe(z.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    n_train = len(x_train)
    best_val_acc, best_state = -1.0, None
    for epoch in range(probe_epochs):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, probe_batch_size):
            idx = perm[i:i + probe_batch_size]
            xb, yb = x_train[idx].to(device), y_train[idx].to(device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_acc = (model(x_val.to(device)).argmax(-1).cpu() == y_val).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x_test.to(device))
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        acc_test = (logits.argmax(-1).cpu() == y_test).float().mean().item()
    try:
        if n_classes > 2:
            auc_test = roc_auc_score(y_test.numpy(), probs, multi_class="ovr", average="macro")
        else:
            auc_test = roc_auc_score(y_test.numpy(), probs[:, 1])
    except ValueError:
        auc_test = float("nan")
    return acc_test, auc_test, best_val_acc


def run_probe(embeddings, labels):
    sm_event_ids = np.where(np.isin(labels, sm_labels))[0]
    label_to_idx = {lab: i for i, lab in enumerate(sorted(sm_labels))}
    n_classes = len(sm_labels)

    train_pool, val_ev, test_ev = make_val_test_split(sm_event_ids)
    sizes = sorted(set([s for s in train_sizes if s <= len(train_pool)] + [len(train_pool)]))

    rows = []
    for field_name, z in embeddings.items():
        for size in sizes:
            for seed in range(n_seeds):
                rng = np.random.default_rng(seed * 100_003 + size)
                train_ev = rng.choice(train_pool, size=size, replace=False) if size < len(train_pool) else train_pool
                acc, auc, val_acc = train_probe(z, labels, label_to_idx, train_ev, val_ev, test_ev, n_classes, seed)
                rows.append(dict(field=field_name, n_train=size, seed=seed, val_acc=val_acc, test_acc=acc, test_auc=auc))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "downstream_probe_results.csv"), index=False)
    print(df.groupby(["field", "n_train"])[["test_acc", "test_auc"]].agg(["mean", "std"]))
    return df


def plot_curves(df):
    colors = {"continuous": "grey", "token": "blue", "vqvae_encoder": "orange"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for metric, ax in zip(["test_acc", "test_auc"], axes):
        for field_name, color in colors.items():
            g = df[df["field"] == field_name].groupby("n_train")[metric].agg(["mean", "std"]).reset_index()
            ax.plot(g["n_train"], g["mean"], marker="o", color=color, label=field_name)
            ax.fill_between(g["n_train"], g["mean"] - g["std"], g["mean"] + g["std"], color=color, alpha=0.2)
        ax.set_xscale("log")
        ax.set_xlabel("number of labeled training events")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "downstream_probe_curves.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    embeddings, labels = run_extract()
    df = run_probe(embeddings, labels)
    plot_curves(df)
