#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cosmetics first-window raw/fusion/Z-only training with safe resume/skip.

Key behavior:
1. Reuses finished runs from a previous results CSV, so raw runs do not need retraining.
2. Uses FIXED Z files by default:
   /content/drive/MyDrive/ltlf_cosmetics_experiments/modelcheck_cosmetics_dec_last20_top20_FIXED
3. Uses explicit top_k for Z.
4. Always aligns Z by session_id_hash and checks labels.
5. Uses new run names for fixed Z/fusion, so old wrong Z results are not reused.

Example:
python /content/train_cosmetics_first_window_fixed_skip.py \
  --previous_results_csv /content/drive/MyDrive/ltlf_cosmetics_experiments/cosmetics_first_window_results.csv \
  --packs B \
  --seeds 42 \
  --skip_existing 1
"""

import argparse
import ast
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def parse_list_cell(x):
    """Parse a dataframe cell that stores a list-like sequence."""
    if isinstance(x, list):
        return [str(t) for t in x]

    if pd.isna(x):
        return []

    s = str(x).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [str(t) for t in v]
    except Exception:
        pass

    # fallback
    for ch in ["[", "]", "'", '"']:
        s = s.replace(ch, " ")
    s = s.replace(",", " ")
    return [t.strip() for t in s.split() if t.strip()]


def parse_prop_trace_cell(x):
    """
    Parse prop_BD / prop_BP / prop_BDP style trace cells into a list of state tokens.
    Each state becomes a joined token, e.g. {'a','dm'} -> 'a|dm'.

    This is for raw Transformer branch only.
    FIXED Z model checking is handled separately and already saved in npz.
    """
    if pd.isna(x):
        return []

    s = str(x).strip()
    try:
        v = ast.literal_eval(s)
    except Exception:
        v = s

    states = []

    if isinstance(v, (list, tuple)):
        for state in v:
            if isinstance(state, (list, tuple, set)):
                toks = sorted([str(t).strip() for t in state if str(t).strip()])
            else:
                ss = str(state)
                for ch in ["{", "}", "[", "]", "(", ")", "'", '"']:
                    ss = ss.replace(ch, " ")
                for sep in ["|", "+", ",", ";", "&"]:
                    ss = ss.replace(sep, " ")
                toks = sorted([t.strip() for t in ss.split() if t.strip()])
            states.append("|".join(toks) if toks else "<EMPTY>")
        return states

    # fallback: treat whole string as one token or semicolon-separated states
    if ";" in s:
        chunks = s.split(";")
        for chunk in chunks:
            toks = chunk.replace(",", " ").replace("|", " ").split()
            toks = sorted([t.strip() for t in toks if t.strip()])
            states.append("|".join(toks) if toks else "<EMPTY>")
        return states

    return [s]


def dwell_sec_to_raw_bin(x):
    """
    Convert raw dwell seconds into coarse numeric bins for the raw baseline.
    This is NOT an LTL/proposition column; it is a simple tokenization of
    the original continuous dwell_sec_sequence so the Transformer can embed it.
    """
    try:
        v = float(x)
    except Exception:
        return "D=NA"

    if not np.isfinite(v):
        return "D=NA"

    if v <= 0:
        return "D=0"
    if v < 1:
        return "D=(0,1)"
    if v < 5:
        return "D=[1,5)"
    if v < 15:
        return "D=[5,15)"
    if v < 60:
        return "D=[15,60)"
    if v < 300:
        return "D=[60,300)"
    return "D>=300"


def pad_to_len(seq, n, fill="<NA>"):
    seq = list(seq)
    if len(seq) >= n:
        return seq[:n]
    return seq + [fill] * (n - len(seq))


def get_raw_tokens(row, pack):
    """
    Raw branch tokens.

    IMPORTANT:
    This function intentionally DOES NOT use prop_B / prop_BD / prop_BP / prop_BDP
    or ltlf_* columns. Those columns are logic/proposition representations and
    must only be used to produce the Z verification branch.

    Raw baselines use original columns:
      B   : behavior_sequence
      BD  : behavior_sequence + dwell_sec_sequence
      BP  : behavior_sequence + product_sequence + category_sequence
      BDP : behavior_sequence + dwell_sec_sequence + product_sequence + category_sequence

    We still convert each timestep into a categorical event token because this
    model is a token-based Transformer. That is tokenization of raw fields, not
    LTL/proposition engineering.
    """
    beh = parse_list_cell(row["behavior_sequence"])

    if pack == "B":
        return [f"A={b}" for b in beh]

    if pack == "BD":
        dwell = parse_list_cell(row["dwell_sec_sequence"])

        n = max(len(beh), len(dwell))
        beh = pad_to_len(beh, n)
        dwell = pad_to_len(dwell, n)

        return [
            f"A={beh[i]}|{dwell_sec_to_raw_bin(dwell[i])}"
            for i in range(n)
        ]

    if pack == "BP":
        prod = parse_list_cell(row["product_sequence"])
        cat = parse_list_cell(row["category_sequence"])

        n = max(len(beh), len(prod), len(cat))
        beh = pad_to_len(beh, n)
        prod = pad_to_len(prod, n)
        cat = pad_to_len(cat, n)

        return [
            f"A={beh[i]}|P={prod[i]}|C={cat[i]}"
            for i in range(n)
        ]

    if pack == "BDP":
        dwell = parse_list_cell(row["dwell_sec_sequence"])
        prod = parse_list_cell(row["product_sequence"])
        cat = parse_list_cell(row["category_sequence"])

        n = max(len(beh), len(dwell), len(prod), len(cat))
        beh = pad_to_len(beh, n)
        dwell = pad_to_len(dwell, n)
        prod = pad_to_len(prod, n)
        cat = pad_to_len(cat, n)

        return [
            f"A={beh[i]}|{dwell_sec_to_raw_bin(dwell[i])}|P={prod[i]}|C={cat[i]}"
            for i in range(n)
        ]

    raise ValueError(f"Unknown raw pack: {pack}. Expected one of B, BD, BP, BDP.")

def truncate_tokens(tokens, raw_len, max_full_len=None):
    """
    raw/fusion uses FIRST events for raw branch.
    """
    if raw_len is None:
        return []

    if str(raw_len).lower() == "full":
        out = tokens
        if max_full_len is not None and len(out) > int(max_full_len):
            out = out[:int(max_full_len)]
        return out

    return tokens[: int(raw_len)]


def find_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


# ============================================================
# Resume / skip handling
# ============================================================

def read_results_csv(path):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def normalize_result_cols(df):
    if df is None or df.empty:
        return df

    df = df.copy()

    if "run" not in df.columns:
        for c in ["name", "run_name", "experiment", "Run"]:
            if c in df.columns:
                df["run"] = df[c].astype(str)
                break

    if "seed" not in df.columns:
        for c in ["Seed"]:
            if c in df.columns:
                df["seed"] = df[c]
                break

    return df


def build_done_set(result_dfs):
    """
    A run is considered done if it has seed + run and at least one non-null metric.
    Handles both F1/MCC/AUPRC/Brier and lowercase columns.

    Also adds raw-run aliases:
      APC10_raw -> BP10_raw
      APCfull_raw -> BPfull_raw
      ADPC10_raw -> BDP10_raw
      ADPCfull_raw -> BDPfull_raw
    so old result files can skip renamed raw experiments.
    """
    done = set()
    raw_alias = {
        "APC10_raw": "BP10_raw",
        "APCfull_raw": "BPfull_raw",
        "ADPC10_raw": "BDP10_raw",
        "ADPCfull_raw": "BDPfull_raw",
    }

    metric_candidates = [
        "F1", "MCC", "AUPRC", "Brier",
        "f1", "mcc", "auprc", "brier",
        "f1_mean", "mcc_mean", "auprc_mean", "brier_mean",
    ]

    for df in result_dfs:
        df = normalize_result_cols(df)
        if df is None or df.empty:
            continue
        if "seed" not in df.columns or "run" not in df.columns:
            continue

        metrics = [c for c in metric_candidates if c in df.columns]

        for _, r in df.iterrows():
            try:
                seed = int(r["seed"])
                run = str(r["run"])
            except Exception:
                continue

            if not metrics:
                done.add((seed, run))
                if run in raw_alias:
                    done.add((seed, raw_alias[run]))
                if run in raw_alias:
                    done.add((seed, raw_alias[run]))
                continue

            ok = False
            for c in metrics:
                try:
                    if not pd.isna(r[c]):
                        ok = True
                        break
                except Exception:
                    pass

            if ok:
                done.add((seed, run))

    return done


def append_result(results_csv, row):
    results_csv = Path(results_csv)
    ensure_dir(results_csv.parent)

    df_row = pd.DataFrame([row])
    header = not results_csv.exists()
    df_row.to_csv(results_csv, mode="a", header=header, index=False)


# ============================================================
# Z loading
# ============================================================

def load_aligned_Z(df, z_dir, pack, top_k, dataset_tag, id_col, label_col):
    z_dir = Path(z_dir)

    exact = z_dir / f"Z_{pack}_top{top_k}_{dataset_tag}.npz"
    fixed_last20 = z_dir / f"Z_{pack}_top{top_k}_{dataset_tag}_last20_FIXED.npz"

    if exact.exists():
        z_path = exact
    elif fixed_last20.exists():
        z_path = fixed_last20
    else:
        # Allow fallback from top20 to top_k slicing.
        fallback = z_dir / f"Z_{pack}_top20_{dataset_tag}.npz"
        fallback_fixed = z_dir / f"Z_{pack}_top20_{dataset_tag}_last20_FIXED.npz"

        if fallback.exists():
            z_path = fallback
        elif fallback_fixed.exists():
            z_path = fallback_fixed
        else:
            raise FileNotFoundError(
                f"Cannot find Z file for pack={pack}, top_k={top_k}. "
                f"Tried: {exact}, {fixed_last20}, {fallback}, {fallback_fixed}"
            )

    npz = np.load(z_path, allow_pickle=True)

    if "Z" not in npz.files:
        raise ValueError(f"{z_path} does not contain key 'Z'")

    Z_all = npz["Z"]
    y_all = npz["y"].astype(int)
    ids_all = npz["session_id_hash"].astype(str)

    if Z_all.shape[1] < top_k:
        raise ValueError(f"{z_path} has only {Z_all.shape[1]} Z columns, requested top_k={top_k}")

    Z_all = Z_all[:, :top_k]

    id_to_idx = pd.Series(np.arange(len(ids_all)), index=ids_all)

    df_ids = df[id_col].astype(str)
    missing = ~df_ids.isin(id_to_idx.index)
    if missing.any():
        raise ValueError(f"Missing {missing.sum()} ids in Z file. Example: {df_ids[missing].head().tolist()}")

    idx = id_to_idx.loc[df_ids].values

    Z = Z_all[idx].astype(np.float32)
    y_z = y_all[idx]
    y_df = df[label_col].astype(int).values

    match = (y_z == y_df).mean()
    if match != 1.0:
        raise ValueError(f"Label mismatch after Z alignment for {z_path}. match={match:.6f}")

    print(f"Loaded aligned Z: {z_path.name}, full_shape={npz['Z'].shape}, used_shape={Z.shape}")

    return Z


# ============================================================
# Dataset
# ============================================================

class CosmeticsDataset(Dataset):
    def __init__(
        self,
        df,
        mode,
        pack,
        vocab=None,
        raw_len=None,
        Z=None,
        label_col="purchase",
        max_full_len=None,
    ):
        self.mode = mode
        self.pack = pack
        self.raw_len = raw_len
        self.label_col = label_col
        self.max_full_len = max_full_len

        self.y = df[label_col].astype(int).values.astype(np.float32)

        self.has_raw = mode in ["raw", "fusion"]
        self.has_z = mode in ["z_only", "fusion"]

        self.Z = Z.astype(np.float32) if Z is not None else None

        if self.has_raw:
            assert vocab is not None
            self.vocab = vocab
            seqs = []
            for _, row in df.iterrows():
                toks = get_raw_tokens(row, pack)
                toks = truncate_tokens(toks, raw_len=raw_len, max_full_len=max_full_len)
                ids = [vocab.get(t, vocab["<UNK>"]) for t in toks]
                if len(ids) == 0:
                    ids = [vocab["<UNK>"]]
                seqs.append(ids)
            self.seqs = seqs
        else:
            self.vocab = None
            self.seqs = None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        item = {"y": torch.tensor(self.y[idx], dtype=torch.float32)}

        if self.has_raw:
            item["x"] = torch.tensor(self.seqs[idx], dtype=torch.long)

        if self.has_z:
            item["z"] = torch.tensor(self.Z[idx], dtype=torch.float32)

        return item


def collate_batch(batch):
    out = {}

    y = torch.stack([b["y"] for b in batch])
    out["y"] = y

    if "x" in batch[0]:
        lengths = torch.tensor([len(b["x"]) for b in batch], dtype=torch.long)
        max_len = int(lengths.max().item())
        x = torch.zeros((len(batch), max_len), dtype=torch.long)
        mask = torch.ones((len(batch), max_len), dtype=torch.bool)  # True = padding

        for i, b in enumerate(batch):
            l = len(b["x"])
            x[i, :l] = b["x"]
            mask[i, :l] = False

        out["x"] = x
        out["mask"] = mask
        out["lengths"] = lengths

    if "z" in batch[0]:
        out["z"] = torch.stack([b["z"] for b in batch])

    return out


def build_vocab(train_df, pack, min_freq=1):
    counts = {}
    for _, row in train_df.iterrows():
        toks = get_raw_tokens(row, pack)
        for t in toks:
            counts[t] = counts.get(t, 0) + 1

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for tok, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if c >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)

    return vocab


def max_len_in_splits(dfs, pack, raw_len, max_full_len=None):
    mx = 1
    for df in dfs:
        for _, row in df.iterrows():
            toks = get_raw_tokens(row, pack)
            toks = truncate_tokens(toks, raw_len=raw_len, max_full_len=max_full_len)
            mx = max(mx, len(toks))
    return mx


# ============================================================
# Models
# ============================================================

class RawTransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.15,
        out_dim=64,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len + 2, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask):
        bsz, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, seq_len)

        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.encoder(h, src_key_padding_mask=mask)

        valid = (~mask).float().unsqueeze(-1)
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        return self.proj(pooled)


class ZEncoder(nn.Module):
    def __init__(self, z_dim, hidden=64, out_dim=64, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.GELU(),
        )

    def forward(self, z):
        return self.net(z)


class CosmeticsPredictor(nn.Module):
    def __init__(
        self,
        mode,
        vocab_size=None,
        max_len=None,
        z_dim=None,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.15,
        hidden=64,
    ):
        super().__init__()
        self.mode = mode

        if mode in ["raw", "fusion"]:
            self.raw_encoder = RawTransformerEncoder(
                vocab_size=vocab_size,
                max_len=max_len,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout,
                out_dim=hidden,
            )
        else:
            self.raw_encoder = None

        if mode in ["z_only", "fusion"]:
            self.z_encoder = ZEncoder(
                z_dim=z_dim,
                hidden=hidden,
                out_dim=hidden,
                dropout=dropout,
            )
        else:
            self.z_encoder = None

        if mode == "raw":
            final_dim = hidden
        elif mode == "z_only":
            final_dim = hidden
        elif mode == "fusion":
            final_dim = hidden * 2
        else:
            raise ValueError(mode)

        self.head = nn.Sequential(
            nn.LayerNorm(final_dim),
            nn.Linear(final_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch):
        feats = []

        if self.mode in ["raw", "fusion"]:
            feats.append(self.raw_encoder(batch["x"], batch["mask"]))

        if self.mode in ["z_only", "fusion"]:
            feats.append(self.z_encoder(batch["z"]))

        h = feats[0] if len(feats) == 1 else torch.cat(feats, dim=1)
        return self.head(h).squeeze(-1)


# ============================================================
# Training / evaluation
# ============================================================

@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    probs = []
    ys = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        y = batch["y"].detach().cpu().numpy()
        probs.append(p)
        ys.append(y)

    return np.concatenate(probs), np.concatenate(ys).astype(int)


def compute_metrics(y, p, threshold=0.5):
    yhat = (p >= threshold).astype(int)

    out = {}
    out["F1"] = f1_score(y, yhat, zero_division=0)
    out["MCC"] = matthews_corrcoef(y, yhat) if len(np.unique(yhat)) > 1 else 0.0
    out["AUPRC"] = average_precision_score(y, p)
    out["Brier"] = brier_score_loss(y, p)

    try:
        out["AUC"] = roc_auc_score(y, p)
    except Exception:
        out["AUC"] = np.nan

    out["threshold"] = threshold
    out["pred_pos_rate"] = float(yhat.mean())
    return out


def best_threshold_by_val(y, p, metric="mcc"):
    grid = np.linspace(0.05, 0.95, 181)
    best_t = 0.5
    best_score = -1e18

    for t in grid:
        yhat = (p >= t).astype(int)

        if metric == "mcc":
            score = matthews_corrcoef(y, yhat) if len(np.unique(yhat)) > 1 else 0.0
        elif metric == "f1":
            score = f1_score(y, yhat, zero_division=0)
        else:
            raise ValueError(metric)

        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t, best_score


def train_one_run(seed, run_cfg, args, device):
    set_seed(seed)

    seed_dir = Path(args.split_dir) / f"seed_{seed}"
    train_df = pd.read_csv(seed_dir / "train.csv")
    val_df = pd.read_csv(seed_dir / "val.csv")
    test_df = pd.read_csv(seed_dir / "test.csv")

    label_col = args.label_col
    id_col = args.id_col

    mode = run_cfg["mode"]
    raw_pack = run_cfg.get("raw_pack")
    z_pack = run_cfg.get("z_pack")
    raw_len = run_cfg.get("raw_len")
    z_top_k_value = run_cfg.get("z_top_k", args.z_top_k)
    z_top_k = args.z_top_k if z_top_k_value is None else int(z_top_k_value)

    print(f"train/val/test: {train_df.shape}, {val_df.shape}, {test_df.shape}")
    print(
        f"positive rates: train={train_df[label_col].mean():.4f}, "
        f"val={val_df[label_col].mean():.4f}, test={test_df[label_col].mean():.4f}"
    )

    Z_train = Z_val = Z_test = None
    if mode in ["z_only", "fusion"]:
        Z_train = load_aligned_Z(
            train_df, args.z_dir, z_pack, z_top_k,
            args.dataset_tag, id_col, label_col
        )
        Z_val = load_aligned_Z(
            val_df, args.z_dir, z_pack, z_top_k,
            args.dataset_tag, id_col, label_col
        )
        Z_test = load_aligned_Z(
            test_df, args.z_dir, z_pack, z_top_k,
            args.dataset_tag, id_col, label_col
        )

        # quick support debug for first Z feature
        ytr = train_df[label_col].astype(int).values
        if Z_train.shape[1] > 0:
            pos = Z_train[ytr == 1, 0].mean()
            neg = Z_train[ytr == 0, 0].mean()
            print(f"Z sanity T01 train support: pos={pos:.4f} neg={neg:.4f} gap={pos-neg:.4f}")

    vocab = None
    max_seq_len = None
    if mode in ["raw", "fusion"]:
        vocab = build_vocab(train_df, raw_pack)
        max_seq_len = max_len_in_splits(
            [train_df, val_df, test_df],
            raw_pack,
            raw_len=raw_len,
            max_full_len=args.max_full_len,
        )
        print(f"raw vocab size={len(vocab)}, max_seq_len={max_seq_len}, raw_len={raw_len}")

    train_ds = CosmeticsDataset(
        train_df,
        mode=mode,
        pack=raw_pack,
        vocab=vocab,
        raw_len=raw_len,
        Z=Z_train,
        label_col=label_col,
        max_full_len=args.max_full_len,
    )
    val_ds = CosmeticsDataset(
        val_df,
        mode=mode,
        pack=raw_pack,
        vocab=vocab,
        raw_len=raw_len,
        Z=Z_val,
        label_col=label_col,
        max_full_len=args.max_full_len,
    )
    test_ds = CosmeticsDataset(
        test_df,
        mode=mode,
        pack=raw_pack,
        vocab=vocab,
        raw_len=raw_len,
        Z=Z_test,
        label_col=label_col,
        max_full_len=args.max_full_len,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )

    model = CosmeticsPredictor(
        mode=mode,
        vocab_size=len(vocab) if vocab is not None else None,
        max_len=max_seq_len if max_seq_len is not None else None,
        z_dim=Z_train.shape[1] if Z_train is not None else None,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        hidden=args.hidden,
    ).to(device)

    y_train = train_df[label_col].astype(int).values
    n_pos = max(1, int((y_train == 1).sum()))
    n_neg = max(1, int((y_train == 0).sum()))
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = -1e18
    best_state = None
    best_epoch = -1
    bad_epochs = 0

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            logits = model(batch)
            loss = criterion(logits, batch["y"])
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        val_p, val_y = predict_probs(model, val_loader, device)

        if args.threshold_mode == "val_mcc":
            val_t, _ = best_threshold_by_val(val_y, val_p, metric="mcc")
        elif args.threshold_mode == "val_f1":
            val_t, _ = best_threshold_by_val(val_y, val_p, metric="f1")
        else:
            val_t = 0.5

        val_metrics = compute_metrics(val_y, val_p, threshold=val_t)
        val_score = val_metrics["MCC"]

        print(
            f"  epoch={epoch:03d} loss={np.mean(losses):.4f} "
            f"val_mcc={val_metrics['MCC']:.4f} val_f1={val_metrics['F1']:.4f} "
            f"val_auprc={val_metrics['AUPRC']:.4f} thr={val_t:.3f}"
        )

        if val_score > best_val:
            best_val = val_score
            best_epoch = epoch
            bad_epochs = 0
            best_state = {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "vocab": vocab,
                "run_cfg": run_cfg,
                "seed": seed,
                "val_threshold": val_t,
                "val_metrics": val_metrics,
            }
        else:
            bad_epochs += 1

        if args.patience > 0 and bad_epochs >= args.patience:
            print(f"  early stopping at epoch={epoch}, best_epoch={best_epoch}")
            break

    train_time = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        final_threshold = float(best_state["val_threshold"])
    else:
        final_threshold = 0.5

    infer_t0 = time.time()
    test_p, test_y = predict_probs(model, test_loader, device)
    infer_time = time.time() - infer_t0

    test_metrics = compute_metrics(test_y, test_p, threshold=final_threshold)

    ckpt_dir = Path(args.out_dir) / "checkpoints" / f"seed_{seed}"
    ensure_dir(ckpt_dir)
    ckpt_path = ckpt_dir / f"{run_cfg['name']}.pt"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "best_state_meta": best_state,
            "test_metrics": test_metrics,
            "args": vars(args),
            "run_cfg": run_cfg,
        },
        ckpt_path,
    )

    row = {
        "seed": seed,
        "run": run_cfg["name"],
        "mode": mode,
        "raw_pack": raw_pack,
        "z_pack": z_pack,
        "raw_len": raw_len,
        "z_top_k": z_top_k if mode in ["z_only", "fusion"] else np.nan,
        "threshold_mode": args.threshold_mode,
        "threshold": test_metrics["threshold"],
        "F1": test_metrics["F1"],
        "MCC": test_metrics["MCC"],
        "AUC": test_metrics["AUC"],
        "AUPRC": test_metrics["AUPRC"],
        "Brier": test_metrics["Brier"],
        "pred_pos_rate": test_metrics["pred_pos_rate"],
        "best_epoch": best_epoch,
        "train_time_sec": train_time,
        "infer_time_sec": infer_time,
        "checkpoint": str(ckpt_path),
        "z_dir": args.z_dir if mode in ["z_only", "fusion"] else "",
    }

    return row


# ============================================================
# Run config
# ============================================================

def make_runs(packs, z_top_k):
    """
    Clean experiment names.

    For each pack:
      - raw first-10
      - raw first-20
      - raw full
      - Z-only topK
      - raw first-10 + Z fusion
      - raw first-20 + Z fusion
      - raw full + Z fusion

    Raw branch uses raw sequence columns only:
      B   : behavior_sequence
      BD  : behavior_sequence + dwell_sec_sequence
      BP  : behavior_sequence + product_sequence + category_sequence
      BDP : behavior_sequence + dwell_sec_sequence + product_sequence + category_sequence

    Z branch uses precomputed fixed model-checking features from z_dir.
    """
    runs = []

    for pack in packs:
        for raw_len in [10, 20, "full"]:
            len_label = "full" if raw_len == "full" else str(raw_len)

            runs.append({
                "name": f"{pack}{len_label}_raw",
                "raw_pack": pack,
                "z_pack": None,
                "raw_len": raw_len,
                "z_top_k": None,
                "mode": "raw",
            })

        runs.append({
            "name": f"Z_{pack}_top{z_top_k}_only",
            "raw_pack": None,
            "z_pack": pack,
            "raw_len": None,
            "z_top_k": z_top_k,
            "mode": "z_only",
        })

        for raw_len in [10, 20, "full"]:
            len_label = "full" if raw_len == "full" else str(raw_len)

            runs.append({
                "name": f"{pack}{len_label}_Z_{pack}_top{z_top_k}",
                "raw_pack": pack,
                "z_pack": pack,
                "raw_len": raw_len,
                "z_top_k": z_top_k,
                "mode": "fusion",
            })

    return runs

def filter_runs(runs, only_runs=None, exclude_runs=None):
    if only_runs:
        wanted = set([x.strip() for x in only_runs.split(",") if x.strip()])
        runs = [r for r in runs if r["name"] in wanted]

    if exclude_runs:
        banned = set([x.strip() for x in exclude_runs.split(",") if x.strip()])
        runs = [r for r in runs if r["name"] not in banned]

    return runs


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base", type=str, default="/content/drive/MyDrive/ltlf_cosmetics_experiments")
    ap.add_argument("--split_dir", type=str, default=None)
    ap.add_argument("--z_dir", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--dataset_tag", type=str, default="cosmetics_dec")

    ap.add_argument("--results_csv", type=str, default=None)
    ap.add_argument("--previous_results_csv", type=str, default=None)
    ap.add_argument("--skip_existing", type=int, default=1)

    ap.add_argument("--packs", nargs="+", default=["B"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--z_top_k", type=int, default=10)
    ap.add_argument("--only_runs", type=str, default=None)
    ap.add_argument("--exclude_runs", type=str, default=None)

    ap.add_argument("--label_col", type=str, default="purchase")
    ap.add_argument("--id_col", type=str, default="session_id_hash")

    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--eval_batch_size", type=int, default=1024)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.15)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--threshold_mode", type=str, default="fixed", choices=["fixed", "val_mcc", "val_f1"])
    ap.add_argument("--max_full_len", type=int, default=None)

    args = ap.parse_args()

    base = Path(args.base)

    if args.split_dir is None:
        args.split_dir = str(base / "cosmetics_dec_ready" / "cosmetics_dec_splits")

    if args.z_dir is None:
        args.z_dir = str(base / "modelcheck_cosmetics_dec_last20_top20_FIXED")

    if args.out_dir is None:
        args.out_dir = str(base / "cosmetics_first_window_training_FIXED")

    if args.results_csv is None:
        args.results_csv = str(Path(args.out_dir) / "cosmetics_first_window_results_FIXED.csv")

    ensure_dir(args.out_dir)

    # If previous_results_csv is omitted, try the common old location.
    inferred_prev = base / "cosmetics_first_window_results.csv"
    prev_csvs = []
    if args.previous_results_csv:
        prev_csvs.append(args.previous_results_csv)
    elif inferred_prev.exists():
        prev_csvs.append(str(inferred_prev))

    result_dfs = [read_results_csv(args.results_csv)]
    result_dfs += [read_results_csv(p) for p in prev_csvs]

    done = build_done_set(result_dfs)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Cosmetics first-window raw/fusion/Z-only training with skip/resume")
    print("=" * 80)
    print("split_dir:", args.split_dir)
    print("z_dir:", args.z_dir)
    print("out_dir:", args.out_dir)
    print("results_csv:", args.results_csv)
    print("previous_results_csv:", prev_csvs)
    print("device:", device)
    print("packs:", args.packs)
    print("seeds:", args.seeds)
    print("z_top_k:", args.z_top_k)
    print("skip_existing:", bool(args.skip_existing))
    print("threshold_mode:", args.threshold_mode)
    print("IMPORTANT: raw/fusion uses FIRST events with tokens[:max_len].")
    print("IMPORTANT: raw branch uses raw sequence columns only, never prop_* or ltlf_* columns.")
    print("IMPORTANT: run names are clean; use a fresh out_dir/results_csv to avoid mixing old wrong Z results.")
    print("done runs detected:", len(done))

    runs = make_runs(args.packs, args.z_top_k)
    runs = filter_runs(runs, only_runs=args.only_runs, exclude_runs=args.exclude_runs)

    print("\nRuns:")
    for r in runs:
        print(" ", r["name"], r)

    for seed in args.seeds:
        print("\n" + "#" * 80)
        print(f"SEED {seed}")
        print("#" * 80)

        for cfg in runs:
            key = (int(seed), str(cfg["name"]))

            if args.skip_existing and key in done:
                print(f"\nSKIP seed={seed} run={cfg['name']} because it already exists in results CSV.")
                continue

            print(f"\nRunning {cfg['name']}: {cfg}")

            try:
                row = train_one_run(seed, cfg, args, device)
                append_result(args.results_csv, row)
                done.add(key)

                print(
                    f"RESULT seed={seed} run={cfg['name']} "
                    f"F1={row['F1']:.4f} MCC={row['MCC']:.4f} "
                    f"AUPRC={row['AUPRC']:.4f} Brier={row['Brier']:.4f} "
                    f"train={row['train_time_sec']:.1f}s"
                )

            except Exception as e:
                print(f"ERROR seed={seed} run={cfg['name']}: {repr(e)}")
                raise

    print("\nDone.")
    print("Results saved to:", args.results_csv)


if __name__ == "__main__":
    main()
