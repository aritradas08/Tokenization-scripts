import os
import math
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

data_path = "/global/cfs/cdirs/m4958/usr/aritra08/Anomaly_Detection/data/combined_pu0_1500k.parquet"
save_dir = "/global/cfs/cdirs/m4958/usr/aritra08/vqvae/8192/results_8192"
raw_cols = ["d0", "z0", "theta", "p", "eta", "phi", "pt"]
feat_names = ["d0_symlog", "z0_symlog", "log_pt", "eta", "cos_dphi", "sin_dphi"]
n_feats = len(feat_names)

sm_labels = [0, 1, 2]
bsm_labels = [3, 4]
bsm_process_names = {3: "Higs Portal", 4: "Hidden Valley"}

max_tracks = 128 # Max no of tracks per event 
val_frac = 0.1
seed = 0

# VQ Hyperparmeters

codebook_size = 8192
code_dim = 8
model_dim = 64
n_heads = 8
n_layers = 4
ffn_dim = 512
dropout = 0.0

ema_decay = 0.99
commit_weight = 1.0
replace_every = 10
replace_until = 0.7

feature_weights = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0] #d0 and z0 get twice the feature importance while calculting the quantization reco error

batch_size = 512
n_epochs = 30
lr = 5e-4
weight_decay = 1e-5
warmup_steps = 500
grad_clip = 1.0
num_workers = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def symexp_t(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))


def prepare_data(path):
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
    phi_ref = np.repeat(cols["phi"][starts], lengths)
    dphi = (cols["phi"] - phi_ref + np.pi) % (2 * np.pi) - np.pi

    feats = np.column_stack([
        symlog(cols["d0"]),
        symlog(cols["z0"]),
        np.log(np.clip(cols["pt"], 1e-3, None)),
        cols["eta"],
        np.cos(dphi),
        np.sin(dphi),
    ]).astype(np.float32)
    
    rank = np.arange(len(ev)) - np.repeat(starts, lengths)
    feats = feats[rank < max_tracks]
    new_len = np.minimum(lengths, max_tracks)
    offsets = np.concatenate([[0], np.cumsum(new_len)]).astype(np.int64)
    return feats, offsets, labels


class TrackSetDataset(Dataset):
    def __init__(self, feats, offsets, event_ids):
        self.feats, self.offsets, self.event_ids = feats, offsets, event_ids

    def __len__(self):
        return len(self.event_ids)

    def __getitem__(self, i):
        e = self.event_ids[i]
        return self.feats[self.offsets[e]:self.offsets[e + 1]], e


def collate(batch):
    xs, evs = zip(*batch)
    b, longest = len(xs), max(x.shape[0] for x in xs)
    x = np.zeros((b, longest, n_feats), dtype=np.float32)
    m = np.zeros((b, longest), dtype=bool)
    for i, a in enumerate(xs):
        x[i, :len(a)] = a
        m[i, :len(a)] = True
    return torch.from_numpy(x), torch.from_numpy(m), torch.as_tensor(np.array(evs), dtype=torch.long)


def make_transformer(dim, heads, layers, ffn, drop):
    layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=ffn, dropout=drop, activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)


# VQ part

class VectorQuantizer(nn.Module):
    def __init__(self, k, d, decay=ema_decay, eps=1e-5):
        super().__init__()
        self.k, self.d, self.decay, self.eps = k, d, decay, eps
        self.register_buffer("codebook", torch.randn(k, d))
        self.register_buffer("cluster_size", torch.ones(k))
        self.register_buffer("embed_sum", torch.zeros(k, d))
        self.register_buffer("usage", torch.zeros(k))
        self.register_buffer("initted", torch.tensor(False))
        self.register_buffer("step", torch.tensor(0))
        self.allow_replace = True

    @torch.no_grad()
    def assign(self, zv, chunk=16384):
        cb = self.codebook
        cb_norm = (cb ** 2).sum(1)
        out = []
        for c in zv.split(chunk):
            dist = (c ** 2).sum(1, keepdim=True) - 2 * c @ cb.T + cb_norm
            out.append(dist.argmin(1))
        return torch.cat(out)

    @torch.no_grad()
    def _init(self, zv):
        if len(zv) < self.k:
            sel = torch.randint(0, len(zv), (self.k,), device=zv.device)
        else:
            sel = torch.randperm(len(zv), device=zv.device)[:self.k]
        self.codebook.copy_(zv[sel])
        self.embed_sum.copy_(self.codebook)
        self.cluster_size.fill_(1.0)
        self.initted.fill_(True)

    @torch.no_grad()
    def _ema_update(self, zv, idx):
        counts = torch.bincount(idx, minlength=self.k).float()
        sums = torch.zeros_like(self.codebook).index_add_(0, idx, zv)
        self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        self.embed_sum.mul_(self.decay).add_(sums, alpha=1 - self.decay)
        total = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.eps) / (total + self.k * self.eps) * total
        self.codebook.copy_(self.embed_sum / smoothed.unsqueeze(1))
        self.usage.add_(counts)

    @torch.no_grad()
    def _replace_dead(self, zv):
        dead = self.usage == 0
        n_dead = int(dead.sum())
        if n_dead > 0 and self.allow_replace:
            sel = torch.randint(0, len(zv), (n_dead,), device=zv.device)
            self.codebook[dead] = zv[sel]
            self.embed_sum[dead] = zv[sel]
            self.cluster_size[dead] = 1.0
        self.usage.zero_()

    def forward(self, z, mask):
        zv = z[mask]
        if self.training and not bool(self.initted):
            self._init(zv.detach())
        idx = self.assign(zv.detach())
        cb_sel = self.codebook[idx]
        qerr_v = ((zv.detach() - cb_sel) ** 2).sum(-1)
        commit = F.mse_loss(zv, cb_sel.detach())

        if self.training:
            self._ema_update(zv.detach(), idx)
            self.step += 1
            if int(self.step) % replace_every == 0:
                self._replace_dead(zv.detach())

        zq_v = zv + (cb_sel - zv).detach()
        zq = torch.zeros_like(z)
        zq[mask] = zq_v
        idx_full = torch.full(mask.shape, -1, dtype=torch.long, device=z.device)
        idx_full[mask] = idx
        qerr = torch.zeros(mask.shape, device=z.device)
        qerr[mask] = qerr_v

        p = torch.bincount(idx, minlength=self.k).float()
        p = p / p.sum()
        perplexity = torch.exp(-(p * torch.log(p + 1e-10)).sum())
        return zq, idx_full, commit, qerr, perplexity


class TrackVQVAE(nn.Module):
    def __init__(self, n_feats=n_feats, codebook_size=codebook_size, code_dim=code_dim, model_dim=model_dim, n_heads=n_heads, n_layers=n_layers, ffn_dim=ffn_dim, dropout=dropout, feature_weights=feature_weights):
        super().__init__()
        self.enc_in = nn.Linear(n_feats, model_dim)
        self.encoder = make_transformer(model_dim, n_heads, n_layers, ffn_dim, dropout)
        self.enc_out = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, code_dim))
        self.vq = VectorQuantizer(codebook_size, code_dim)
        self.dec_in = nn.Linear(code_dim, model_dim)
        self.decoder = make_transformer(model_dim, n_heads, n_layers, ffn_dim, dropout)
        self.dec_out = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, n_feats))
        w = torch.tensor(feature_weights, dtype=torch.float32)
        self.register_buffer("feat_w", w / w.mean())

    def encode(self, x, mask):
        h = self.encoder(self.enc_in(x), src_key_padding_mask=~mask)
        return self.enc_out(h)

    def decode(self, zq, mask):
        h = self.decoder(self.dec_in(zq), src_key_padding_mask=~mask)
        return self.dec_out(h)

    def forward(self, x, mask):
        z = self.encode(x, mask)
        zq, idx, commit, qerr, ppl = self.vq(z, mask)
        xr = self.decode(zq, mask)
        return xr, idx, commit, qerr, ppl, z

    def recon_error(self, xr, x):
        return (((xr - x) ** 2) * self.feat_w).mean(-1)


def model_cfg():
    return dict(codebook_size=codebook_size, code_dim=code_dim, model_dim=model_dim, n_heads=n_heads, n_layers=n_layers, ffn_dim=ffn_dim, dropout=dropout, feature_weights=feature_weights)


def train():
    os.makedirs(save_dir, exist_ok=True)
    feats, offsets, labels = prepare_data(data_path)
    n_events = len(labels)

    # SM events split into train and val
    sm_ev = np.where(np.isin(labels, sm_labels))[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(sm_ev)
    n_val = int(val_frac * len(sm_ev))
    val_ev, train_ev = np.sort(sm_ev[:n_val]), np.sort(sm_ev[n_val:])

    # Scaler is fit on train sm tracks only, since we have been pretraining on SM tracks for all our studies
    is_train = np.zeros(n_events, dtype=bool)
    is_train[train_ev] = True
    track_ev = np.repeat(np.arange(n_events), np.diff(offsets))
    scaler = StandardScaler().fit(feats[is_train[track_ev]])
    del track_ev
    feats = scaler.transform(feats).astype(np.float32)

    with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    np.savez(os.path.join(save_dir, "split.npz"), train_ev=train_ev, val_ev=val_ev)

    train_loader = DataLoader(TrackSetDataset(feats, offsets, train_ev), batch_size=batch_size, shuffle=True,
                              collate_fn=collate, drop_last=True, num_workers=num_workers,
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(TrackSetDataset(feats, offsets, val_ev), batch_size=batch_size, shuffle=False,
                            collate_fn=collate, num_workers=num_workers)

    cfg = model_cfg()
    model = TrackVQVAE(**cfg).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    total_steps = n_epochs * len(train_loader)
    lr_fn = lambda s: min(1.0, (s + 1) / warmup_steps) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    history, best_val = [], float("inf")
    for epoch in range(1, n_epochs + 1):
        model.train()
        model.vq.allow_replace = epoch <= replace_until * n_epochs
        agg = dict(recon=0.0, commit=0.0, ppl=0.0, n=0)
        for x, m, _ in train_loader:
            x, m = x.to(device), m.to(device)
            xr, idx, commit, qerr, ppl, _ = model(x, m)
            recon = model.recon_error(xr, x)[m].mean()
            loss = recon + commit_weight * commit
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()
            agg["recon"] += recon.item()
            agg["commit"] += commit.item()
            agg["ppl"] += ppl.item()
            agg["n"] += 1

        # validation on sm events
        model.eval()
        v_recon, v_n = 0.0, 0
        used = torch.zeros(cfg["codebook_size"], dtype=torch.bool, device=device)
        with torch.no_grad():
            for x, m, _ in val_loader:
                x, m = x.to(device), m.to(device)
                xr, idx, commit, qerr, ppl, _ = model(x, m)
                e = model.recon_error(xr, x)[m]
                v_recon += e.sum().item()
                v_n += e.numel()
                used[idx[m]] = True
        v_recon /= max(v_n, 1)

        n = agg["n"]
        history.append(dict(epoch=epoch, train_recon=agg["recon"] / n, train_commit=agg["commit"] / n,
                            train_perplexity=agg["ppl"] / n, val_recon=v_recon,
                            val_codebook_usage=used.float().mean().item(), lr=sched.get_last_lr()[0]))
        print("epoch", epoch, "train recon", history[-1]["train_recon"], "val recon", v_recon, flush=True)

        if v_recon < best_val:
            best_val = v_recon
            torch.save(dict(state_dict=model.state_dict(), cfg=cfg, epoch=epoch), os.path.join(save_dir, "vqvae_best.pt"))
        pd.DataFrame(history).to_csv(os.path.join(save_dir, "training_history_vqvae.csv"), index=False)

    torch.save(dict(state_dict=model.state_dict(), cfg=cfg, epoch=n_epochs), os.path.join(save_dir, "vqvae_last.pt"))


def wrap_angle(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def physical_residuals(x_scaled, xr_scaled, mean, scale):   # reco-original for d0, z0, relative pt, eta, dphi
    xe, re = x_scaled * scale + mean, xr_scaled * scale + mean
    out = [
        symexp_t(re[:, 0]) - symexp_t(xe[:, 0]),
        symexp_t(re[:, 1]) - symexp_t(xe[:, 1]),
        torch.exp(re[:, 2] - xe[:, 2]) - 1.0,
        re[:, 3] - xe[:, 3],
        wrap_angle(torch.atan2(re[:, 5], re[:, 4]) - torch.atan2(xe[:, 5], xe[:, 4])),
    ]
    return torch.stack(out, dim=1)


resid_names = ["d0", "z0", "pt_rel", "eta", "dphi"]

group_names = {1: "sm_val", 2: bsm_process_names.get(3, "bsm_3"), 3: bsm_process_names.get(4, "bsm_4")}


@torch.no_grad()
def extract(max_resid_events=20000):
    ckpt = torch.load(os.path.join(save_dir, "vqvae_best.pt"), map_location=device)
    model = TrackVQVAE(**ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with open(os.path.join(save_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    val_ev = np.load(os.path.join(save_dir, "split.npz"))["val_ev"]

    feats, offsets, labels = prepare_data(data_path)
    feats = scaler.transform(feats).astype(np.float32)
    n_events, k, d = len(labels), ckpt["cfg"]["codebook_size"], ckpt["cfg"]["code_dim"]

    rng = np.random.default_rng(seed)
    grp = np.zeros(n_events, dtype=np.int8)
    grp[rng.permutation(val_ev)[:max_resid_events]] = 1
    bsm3_ev = np.where(labels == 3)[0]
    bsm4_ev = np.where(labels == 4)[0]
    grp[rng.permutation(bsm3_ev)[:max_resid_events]] = 2
    grp[rng.permutation(bsm4_ev)[:max_resid_events]] = 3
    grp_t = torch.from_numpy(grp).to(device)

    tokens = np.zeros(len(feats), dtype=np.int32)
    ev_recon, ev_qerr = np.zeros(n_events, np.float32), np.zeros(n_events, np.float32)
    z_mean = np.zeros((n_events, d), np.float32)
    resid = {1: [], 2: [], 3: []}
    mean_t = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    scale_t = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)

    loader = DataLoader(TrackSetDataset(feats, offsets, np.arange(n_events)), batch_size=batch_size, shuffle=False,
                        collate_fn=collate, num_workers=num_workers)
    for x, m, ev in loader:
        x, m = x.to(device), m.to(device)
        xr, idx, _, qerr, _, z = model(x, m)
        mf = m.float()
        n = mf.sum(1)
        ev_np = ev.numpy()
        ev_recon[ev_np] = ((model.recon_error(xr, x) * mf).sum(1) / n).cpu().numpy()
        ev_qerr[ev_np] = ((qerr * mf).sum(1) / n).cpu().numpy()
        z_mean[ev_np] = ((z * mf[..., None]).sum(1) / n[:, None]).cpu().numpy()
        tokens[offsets[ev_np[0]]:offsets[ev_np[-1] + 1]] = idx[m].cpu().numpy()  # events are consecutive
        g = grp_t[ev.to(device)]
        for gi in (1, 2, 3):
            sel = m & (g == gi)[:, None]
            if sel.any():
                resid[gi].append(physical_residuals(x[sel], xr[sel], mean_t, scale_t).cpu())

    n_tracks = np.diff(offsets)

    np.savez(os.path.join(save_dir, "tokens_all_events.npz"), tokens=tokens, offsets=offsets, labels=labels,
             n_tracks=n_tracks, ev_recon_err=ev_recon, ev_quant_err=ev_qerr, z_mean=z_mean,
             codebook=model.vq.codebook.cpu().numpy())


if __name__ == "__main__":
    train()
    extract()
