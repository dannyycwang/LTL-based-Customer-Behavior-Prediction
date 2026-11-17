#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Product/category raw-ID vs LTLf verification experiment for clickstream prediction.

This script is designed to be used with the previous integrated experiment file
(e.g., ltl_integrated_pytorch_behavior_dwell_automaton_experiment.py). It reuses
that file's LTLf evaluator, metrics, action-only/dwell fusion models, and output style,
while adding raw product/category ID Transformer branches.

Core principle:
  - Raw neural baselines may use raw product_sku_hash_sequence and category_hash_sequence.
  - Raw neural baselines MUST NOT use product/category relation atoms such as fp/rp/sp/pc/p2/p3/sc/cc/rc.
  - Relation atoms enter only through LTLf verification features Z.

Typical four runs:
  no_dwell top10
  no_dwell top20
  with_dwell top10
  with_dwell top20
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, choices=["no_dwell", "with_dwell"],
                        help="Experiment config. no_dwell uses product/category LTL formulas without dwell atoms; with_dwell uses compact formulas with dwell atoms.")
    parser.add_argument("--top_k", type=int, required=True, choices=[10, 20])
    parser.add_argument("--run_name", type=str, default="", help="Optional output run name. If empty, generated from config/top_k.")

    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--raw_lengths", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--include_full_raw", type=int, default=1)

    parser.add_argument("--base_script", type=str, default="/content/ltl_integrated_pytorch_behavior_dwell_automaton_experiment.py")
    parser.add_argument("--product_compact_csv", type=str, default="/content/browsing_draw_random4_sessions_product_compact.csv")
    parser.add_argument("--base_aligned_splits", type=str, default="/content/drive/MyDrive/data/aligned_splits_v1")
    parser.add_argument("--product_aligned_splits", type=str, default="/content/drive/MyDrive/data/product_tltlf/product_experiment_aligned_splits_v1")
    parser.add_argument("--out_dir", type=str, default="/content/drive/MyDrive/data/product_tltlf/product_rawid_vs_ltl_results")

    parser.add_argument("--seq_col", type=str, default="behavior_sequence")
    parser.add_argument("--label_col", type=str, default="purchase")
    parser.add_argument("--session_col", type=str, default="session_id_hash")
    parser.add_argument("--dwell_seconds_col", type=str, default="dwell_seconds_sequence")
    parser.add_argument("--product_col", type=str, default="product_sku_hash_sequence")
    parser.add_argument("--category_col", type=str, default="category_hash_sequence")

    parser.add_argument("--no_dwell_formula_pattern", type=str,
                        default="/content/drive/MyDrive/data/product_tltlf/behavior_product_category_no_dwell_v1/seed_{seed}/window_last30/formulas/top20_semantic_unique_formulas.json")
    parser.add_argument("--with_dwell_formula_pattern", type=str,
                        default="/content/drive/MyDrive/data/product_tltlf/product_compact_v1/seed_{seed}/window_last30/formulas/top20_semantic_unique_formulas.json")

    parser.add_argument("--mc_window", type=int, default=0, help="0 means full-session model checking.")
    parser.add_argument("--mc_truncation", type=str, default="last", choices=["last", "first"])
    parser.add_argument("--raw_truncation", type=str, default="last", choices=["last", "first"])

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mlp_hidden_dim", type=int, default=64)
    parser.add_argument("--ltl_hidden_dim", type=int, default=64)
    parser.add_argument("--fusion_hidden_dim", type=int, default=128)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=50000)
    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


# ============================================================
# General utilities
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def parse_seq(x):
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if pd.isna(x):
        return []

    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return []

    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple, np.ndarray)):
            return list(obj)
    except Exception:
        pass

    s = s.strip("[](){}<>")
    if not s:
        return []
    return [p.strip().strip("'").strip('"') for p in s.split(",") if p.strip()]


def pad_to_len(seq, n, fill):
    seq = list(seq)
    if len(seq) < n:
        seq = seq + [fill] * (n - len(seq))
    return seq[:n]


def truncate(seq, max_len, mode="last"):
    if max_len <= 0:
        return list(seq)
    if len(seq) <= max_len:
        return list(seq)
    if mode == "first":
        return list(seq[:max_len])
    return list(seq[-max_len:])


def safe_str_token(x, missing):
    if x is None:
        return missing
    s = str(x).strip().strip("'").strip('"')
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return missing
    return s


def json_dumps_trace(x):
    return json.dumps(x, ensure_ascii=False)


# ============================================================
# Dataset / model classes for raw product/category ID branches
# ============================================================

class APCDataset(Dataset):
    def __init__(self, a, p, c, y):
        self.a = torch.as_tensor(a, dtype=torch.long)
        self.p = torch.as_tensor(p, dtype=torch.long)
        self.c = torch.as_tensor(c, dtype=torch.long)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.a[idx], self.p[idx], self.c[idx], self.y[idx]


class ADPCDataset(Dataset):
    def __init__(self, a, d, p, c, y):
        self.a = torch.as_tensor(a, dtype=torch.long)
        self.d = torch.as_tensor(d, dtype=torch.float32)
        self.p = torch.as_tensor(p, dtype=torch.long)
        self.c = torch.as_tensor(c, dtype=torch.long)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.a[idx], self.d[idx], self.p[idx], self.c[idx], self.y[idx]


class APCFusionDataset(Dataset):
    def __init__(self, a, p, c, z, y):
        self.a = torch.as_tensor(a, dtype=torch.long)
        self.p = torch.as_tensor(p, dtype=torch.long)
        self.c = torch.as_tensor(c, dtype=torch.long)
        self.z = torch.as_tensor(z, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.a[idx], self.p[idx], self.c[idx], self.z[idx], self.y[idx]


class ADPCFusionDataset(Dataset):
    def __init__(self, a, d, p, c, z, y):
        self.a = torch.as_tensor(a, dtype=torch.long)
        self.d = torch.as_tensor(d, dtype=torch.float32)
        self.p = torch.as_tensor(p, dtype=torch.long)
        self.c = torch.as_tensor(c, dtype=torch.long)
        self.z = torch.as_tensor(z, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.a[idx], self.d[idx], self.p[idx], self.c[idx], self.z[idx], self.y[idx]


class APCTransformer(nn.Module):
    def __init__(self, action_vocab_size, product_vocab_size, category_vocab_size, max_len,
                 use_dwell=False, d_model=64, n_heads=4, n_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.use_dwell = use_dwell
        self.action_emb = nn.Embedding(action_vocab_size, d_model, padding_idx=0)
        self.product_emb = nn.Embedding(product_vocab_size, d_model, padding_idx=0)
        self.category_emb = nn.Embedding(category_vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)

        if use_dwell:
            self.dwell_proj = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Dropout(dropout))

        self.input_norm = nn.LayerNorm(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def encode(self, a, p, c, d=None):
        bsz, seq_len = a.shape
        pos = torch.arange(seq_len, device=a.device).unsqueeze(0).expand(bsz, seq_len)
        mask = a.eq(0)
        h = self.action_emb(a) + self.product_emb(p) + self.category_emb(c) + self.pos_emb(pos)
        if self.use_dwell:
            h = h + self.dwell_proj(d.unsqueeze(-1))
        h = self.input_norm(h)
        h = self.encoder(h, src_key_padding_mask=mask)
        h = self.norm(h)
        valid = (~mask).float().unsqueeze(-1)
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return pooled

    def forward(self, a, p, c, d=None):
        return self.head(self.encode(a, p, c, d)).squeeze(-1)


class APCLateFusionTransformer(nn.Module):
    def __init__(self, action_vocab_size, product_vocab_size, category_vocab_size, max_len, z_dim,
                 use_dwell=False, d_model=64, n_heads=4, n_layers=2, dim_feedforward=128,
                 ltl_hidden_dim=64, fusion_hidden_dim=128, dropout=0.1):
        super().__init__()
        self.raw_encoder = APCTransformer(
            action_vocab_size=action_vocab_size,
            product_vocab_size=product_vocab_size,
            category_vocab_size=category_vocab_size,
            max_len=max_len,
            use_dwell=use_dwell,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.ltl_encoder = nn.Sequential(
            nn.Linear(z_dim, ltl_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ltl_hidden_dim, ltl_hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model + ltl_hidden_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, a, p, c, z, d=None):
        h_raw = self.raw_encoder.encode(a, p, c, d)
        h_ltl = self.ltl_encoder(z)
        return self.head(torch.cat([h_raw, h_ltl], dim=1)).squeeze(-1)


# ============================================================
# Main runner
# ============================================================

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    base_script = Path(args.base_script)
    if not base_script.exists():
        alt = Path("/content/ltl_integrated_pytorch_behavior_dwell_automaton_experiment(1).py")
        if alt.exists():
            base_script = alt
    assert base_script.exists(), f"Cannot find base script: {base_script}"

    product_compact_csv = Path(args.product_compact_csv)
    if not product_compact_csv.exists():
        alt = Path("/content/drive/MyDrive/data/product_tltlf/product_compact_v1/browsing_draw_random4_sessions_product_compact.csv")
        if alt.exists():
            product_compact_csv = alt
    assert product_compact_csv.exists(), f"Cannot find product compact CSV: {product_compact_csv}"

    base_aligned_splits = Path(args.base_aligned_splits)
    if not base_aligned_splits.exists():
        alt = Path("/content/aligned_splits_v1")
        if alt.exists():
            base_aligned_splits = alt
    assert base_aligned_splits.exists(), f"Cannot find aligned splits: {base_aligned_splits}"

    product_aligned_splits = Path(args.product_aligned_splits)
    out_root = Path(args.out_dir)

    if not args.run_name:
        run_name = f"pc_{args.config}_rawid_vs_ltl_fullcheck_top{args.top_k}"
    else:
        run_name = args.run_name
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import base script as module.
    spec = importlib.util.spec_from_file_location("base_exp", str(base_script))
    base_exp = importlib.util.module_from_spec(spec)
    sys.modules["base_exp"] = base_exp
    spec.loader.exec_module(base_exp)

    print("BASE_SCRIPT:", base_script)
    print("PRODUCT_COMPACT_CSV:", product_compact_csv)
    print("BASE_ALIGNED_SPLITS:", base_aligned_splits)
    print("PRODUCT_ALIGNED_SPLITS:", product_aligned_splits)
    print("OUT_DIR:", out_dir)
    print("CONFIG:", args.config, "TOP_K:", args.top_k, "SEEDS:", args.seeds)
    print("DEVICE:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    def parse_logic_trace(x):
        return base_exp.parse_logic_trace_cell(x)

    def to_no_dwell_trace_from_compact(x):
        keep = {"b", "d", "a", "r", "fp", "rp", "sp", "pc", "p2", "p3", "sc", "cc", "rc"}
        trace = parse_logic_trace(x)
        out = []
        for state in trace:
            new_state = []
            for atom in state:
                atom = str(atom).strip().lower()
                if atom in keep and atom not in new_state:
                    new_state.append(atom)
            out.append(new_state)
        return out

    def ensure_no_dwell_column(df):
        if "logic_trace_no_dwell" not in df.columns:
            if "logic_trace_compact" in df.columns:
                df["logic_trace_no_dwell"] = df["logic_trace_compact"].apply(lambda x: json_dumps_trace(to_no_dwell_trace_from_compact(x)))
            elif "logic_trace" in df.columns:
                df["logic_trace_no_dwell"] = df["logic_trace"].apply(lambda x: json_dumps_trace(to_no_dwell_trace_from_compact(x)))
            else:
                raise KeyError("Cannot create logic_trace_no_dwell")
        return df

    def prepare_product_aligned_splits():
        product_aligned_splits.mkdir(parents=True, exist_ok=True)
        extra_df = pd.read_csv(product_compact_csv).copy()
        extra_df = ensure_no_dwell_column(extra_df)

        needed_cols = [
            args.session_col,
            args.product_col,
            args.category_col,
            args.dwell_seconds_col,
            "logic_trace_compact",
            "logic_trace_no_dwell",
        ]
        missing = [c for c in needed_cols if c not in extra_df.columns]
        assert not missing, f"Missing columns in compact CSV: {missing}"
        extra_df = extra_df[needed_cols].drop_duplicates(subset=[args.session_col])

        for seed in args.seeds:
            in_seed_dir = base_aligned_splits / f"seed_{seed}"
            out_seed_dir = product_aligned_splits / f"seed_{seed}"
            out_seed_dir.mkdir(parents=True, exist_ok=True)
            for split in ["train", "val", "test"]:
                in_path = in_seed_dir / f"{split}.csv"
                out_path = out_seed_dir / f"{split}.csv"
                assert in_path.exists(), f"Missing split: {in_path}"
                df = pd.read_csv(in_path).copy()
                drop_cols = [c for c in needed_cols if c in df.columns and c != args.session_col]
                df = df.drop(columns=drop_cols, errors="ignore")
                df = df.merge(extra_df, on=args.session_col, how="left")
                if args.seq_col not in df.columns:
                    if "action_sequence" in df.columns:
                        df[args.seq_col] = df["action_sequence"]
                    else:
                        raise KeyError(f"Missing {args.seq_col} and action_sequence in {in_path}")
                df.to_csv(out_path, index=False)
            print("Prepared:", out_seed_dir)

    prepare_product_aligned_splits()

    def normalize_action(tok):
        return base_exp.normalize_token(tok, base_exp.DEFAULT_ACTION_TOKEN_MAP)

    def parse_modalities(row):
        raw_actions = parse_seq(row[args.seq_col])
        raw_products = parse_seq(row[args.product_col]) if args.product_col in row.index else []
        raw_categories = parse_seq(row[args.category_col]) if args.category_col in row.index else []
        raw_dwells = base_exp.parse_numeric_sequence(row[args.dwell_seconds_col]) if args.dwell_seconds_col in row.index else []

        n = len(raw_actions)
        raw_products = pad_to_len(raw_products, n, "P0")
        raw_categories = pad_to_len(raw_categories, n, "C0")
        raw_dwells = pad_to_len(raw_dwells, n, 0.0)

        actions, products, categories, dwells = [], [], [], []
        for a, p, c, d in zip(raw_actions, raw_products, raw_categories, raw_dwells):
            norm_a = normalize_action(a)
            if norm_a in {"p", "purchase", "5"}:
                continue
            if not norm_a:
                continue
            actions.append(norm_a)
            products.append(safe_str_token(p, "P0"))
            categories.append(safe_str_token(c, "C0"))
            try:
                fv = float(d)
                if not np.isfinite(fv):
                    fv = 0.0
            except Exception:
                fv = 0.0
            dwells.append(fv)
        return actions, products, categories, dwells

    def build_vocab_from_train(seqs, add_unk=True):
        tokens = sorted({str(t) for seq in seqs for t in seq})
        vocab = {"<PAD>": 0}
        if add_unk:
            vocab["<UNK>"] = 1
        for tok in tokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)
        return vocab

    def encode_id_sequence(seq, vocab, max_len, truncation):
        sub = truncate(seq, max_len, truncation)
        arr = np.zeros(max_len, dtype=np.int64)
        unk = vocab.get("<UNK>", 1)
        for i, tok in enumerate(sub):
            arr[i] = vocab.get(str(tok), unk)
        return arr

    def build_id_matrix(seqs, vocab, max_len, truncation):
        return np.stack([encode_id_sequence(s, vocab, max_len, truncation) for s in seqs])

    def fit_dwell_scaler(seqs, max_len, truncation, log1p=True):
        vals = []
        for seq in seqs:
            for v in truncate(seq, max_len, truncation):
                fv = float(v)
                if log1p:
                    fv = np.log1p(max(fv, 0.0))
                vals.append(fv)
        if not vals:
            return 0.0, 1.0
        vals = np.asarray(vals, dtype=np.float32)
        return float(vals.mean()), float(vals.std() + 1e-8)

    def build_dwell_matrix(seqs, max_len, truncation, mean, std, log1p=True):
        out = []
        for seq in seqs:
            sub = truncate(seq, max_len, truncation)
            arr = np.zeros(max_len, dtype=np.float32)
            for i, v in enumerate(sub):
                fv = float(v)
                if log1p:
                    fv = np.log1p(max(fv, 0.0))
                arr[i] = (fv - mean) / max(std, 1e-8)
            out.append(arr)
        return np.stack(out).astype(np.float32)

    def load_seed_data(seed, logic_trace_col):
        frames = []
        for split in ["train", "val", "test"]:
            path = product_aligned_splits / f"seed_{seed}" / f"{split}.csv"
            assert path.exists(), f"Missing split: {path}"
            df = pd.read_csv(path).copy()
            action_seqs, product_seqs, category_seqs, dwell_seqs, logic_traces = [], [], [], [], []
            for _, row in df.iterrows():
                a, p, c, d = parse_modalities(row)
                action_seqs.append(a)
                product_seqs.append(p)
                category_seqs.append(c)
                dwell_seqs.append(d)
                if logic_trace_col in row.index and not pd.isna(row[logic_trace_col]):
                    logic_traces.append(parse_logic_trace(row[logic_trace_col]))
                else:
                    raise KeyError(f"Missing logic trace col: {logic_trace_col}")
            df["_split"] = split
            df["_action_seq"] = action_seqs
            df["_product_seq"] = product_seqs
            df["_category_seq"] = category_seqs
            df["_dwell_seq"] = dwell_seqs
            df["_logic_trace"] = logic_traces
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    @torch.no_grad()
    def predict_model(model, loader, mode):
        model.eval()
        ys, probs = [], []
        for batch in loader:
            if mode == "z":
                x, y = batch
                x = x.to(device)
                logits = model(x)
            elif mode == "action_ltl":
                a, z, y = batch
                a, z = a.to(device), z.to(device)
                logits = model(a, z)
            elif mode == "action_dwell_ltl":
                a, d, z, y = batch
                a, d, z = a.to(device), d.to(device), z.to(device)
                logits = model(a, d, z)
            elif mode == "apc":
                a, p, c, y = batch
                a, p, c = a.to(device), p.to(device), c.to(device)
                logits = model(a, p, c)
            elif mode == "adpc":
                a, d, p, c, y = batch
                a, d, p, c = a.to(device), d.to(device), p.to(device), c.to(device)
                logits = model(a, p, c, d)
            elif mode == "apc_ltl":
                a, p, c, z, y = batch
                a, p, c, z = a.to(device), p.to(device), c.to(device), z.to(device)
                logits = model(a, p, c, z)
            elif mode == "adpc_ltl":
                a, d, p, c, z, y = batch
                a, d, p, c, z = a.to(device), d.to(device), p.to(device), c.to(device), z.to(device)
                logits = model(a, p, c, z, d)
            else:
                raise ValueError(mode)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            ys.append(y.numpy())
        return np.concatenate(ys), np.concatenate(probs)

    def train_model(model, train_loader, val_loader, test_loader, y_train, mode):
        model = model.to(device)
        pos = float(np.sum(y_train))
        neg = float(len(y_train) - pos)
        pos_weight_value = neg / max(pos, 1.0)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_state, best_val_score, best_epoch = None, -1.0, -1
        patience_counter = 0
        t0 = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                if mode == "z":
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                elif mode == "action_ltl":
                    a, z, y = batch
                    a, z, y = a.to(device), z.to(device), y.to(device)
                    logits = model(a, z)
                elif mode == "action_dwell_ltl":
                    a, d, z, y = batch
                    a, d, z, y = a.to(device), d.to(device), z.to(device), y.to(device)
                    logits = model(a, d, z)
                elif mode == "apc":
                    a, p, c, y = batch
                    a, p, c, y = a.to(device), p.to(device), c.to(device), y.to(device)
                    logits = model(a, p, c)
                elif mode == "adpc":
                    a, d, p, c, y = batch
                    a, d, p, c, y = a.to(device), d.to(device), p.to(device), c.to(device), y.to(device)
                    logits = model(a, p, c, d)
                elif mode == "apc_ltl":
                    a, p, c, z, y = batch
                    a, p, c, z, y = a.to(device), p.to(device), c.to(device), z.to(device), y.to(device)
                    logits = model(a, p, c, z)
                elif mode == "adpc_ltl":
                    a, d, p, c, z, y = batch
                    a, d, p, c, z, y = a.to(device), d.to(device), p.to(device), c.to(device), z.to(device), y.to(device)
                    logits = model(a, p, c, z, d)
                else:
                    raise ValueError(mode)

                loss = criterion(logits, y)
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            y_val, p_val = predict_model(model, val_loader, mode)
            val_metrics = base_exp.compute_metrics(y_val, p_val, threshold=0.5)
            val_score = val_metrics["f1"]
            if val_score > best_val_score:
                best_val_score = float(val_score)
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= args.patience:
                break

        train_time_sec = time.perf_counter() - t0
        if best_state is not None:
            model.load_state_dict(best_state)
            model = model.to(device)

        y_val, p_val = predict_model(model, val_loader, mode)
        threshold, val_threshold_score = base_exp.choose_threshold(y_val, p_val, metric="f1")
        t1 = time.perf_counter()
        y_test, p_test = predict_model(model, test_loader, mode)
        infer_time_sec = time.perf_counter() - t1

        metrics = base_exp.compute_metrics(y_test, p_test, threshold=threshold)
        metrics.update({
            "best_epoch": int(best_epoch),
            "epochs_ran": int(best_epoch if best_epoch > 0 else args.epochs),
            "best_val_score": float(best_val_score),
            "val_threshold_score": float(val_threshold_score),
            "train_time_sec": float(train_time_sec),
            "infer_time_sec": float(infer_time_sec),
            "n_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "device": str(device),
            "pos_weight": float(pos_weight_value),
        })
        return metrics

    def formula_path(seed):
        pattern = args.no_dwell_formula_pattern if args.config == "no_dwell" else args.with_dwell_formula_pattern
        return Path(pattern.format(seed=seed))

    if args.config == "no_dwell":
        logic_trace_col = "logic_trace_no_dwell"
        planned_flags = {
            "verification_only": True,
            "action_ltl": True,
            "action_dwell_ltl": False,
            "apc_raw": True,
            "apc_ltl": True,
            "adpc_raw": False,
            "adpc_ltl": False,
        }
    else:
        logic_trace_col = "logic_trace_compact"
        planned_flags = {
            "verification_only": True,
            "action_ltl": False,
            "action_dwell_ltl": True,
            "apc_raw": False,
            "apc_ltl": False,
            "adpc_raw": True,
            "adpc_ltl": True,
        }

    all_rows = []
    all_mc_rows = []

    for seed in args.seeds:
        set_seed(seed)
        fjson = formula_path(seed)
        if not fjson.exists():
            print(f"Skip seed {seed}: formula JSON not found: {fjson}")
            continue

        print("\n" + "=" * 100)
        print("Seed:", seed)
        print("Formula:", fjson)
        print("=" * 100)

        seed_df = load_seed_data(seed, logic_trace_col)
        train_df = seed_df[seed_df["_split"] == "train"].copy()
        val_df = seed_df[seed_df["_split"] == "val"].copy()
        test_df = seed_df[seed_df["_split"] == "test"].copy()

        y_train = train_df[args.label_col].astype(int).to_numpy()
        y_val = val_df[args.label_col].astype(int).to_numpy()
        y_test = test_df[args.label_col].astype(int).to_numpy()

        action_vocab = base_exp.build_action_vocab(train_df["_action_seq"].tolist())
        product_vocab = build_vocab_from_train(train_df["_product_seq"].tolist(), add_unk=True)
        category_vocab = build_vocab_from_train(train_df["_category_seq"].tolist(), add_unk=True)

        full_len = max(len(s) for s in seed_df["_action_seq"].tolist())
        lengths = list(args.raw_lengths)
        if args.include_full_raw:
            lengths.append(full_len)

        formula_items, trees = base_exp.load_formula_records(fjson, top_k=args.top_k)
        mc_args = SimpleNamespace(
            mc_window=args.mc_window,
            mc_truncation=args.mc_truncation,
            progress_every=args.progress_every,
            ltl_representation="binary",
            max_states_per_formula=64,
        )
        z_all, mc_time, feature_names, representation_meta = base_exp.compute_ltl_representation_from_trees(
            seed_df["_logic_trace"].tolist(), trees, mc_args
        )
        z_train = z_all[seed_df["_split"].values == "train"]
        z_val = z_all[seed_df["_split"].values == "val"]
        z_test = z_all[seed_df["_split"].values == "test"]

        seed_out_dir = out_dir / f"seed_{seed}"
        seed_out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            seed_out_dir / f"Z_seed{seed}_{run_name}.npz",
            Z=z_all.astype(np.float32),
            y=seed_df[args.label_col].astype(int).to_numpy(),
            split=seed_df["_split"].astype(str).to_numpy(),
            session_id_hash=seed_df[args.session_col].astype(str).to_numpy(),
            formula_json=str(fjson),
            model_checking_time_sec=np.array([mc_time]),
            feature_names=np.asarray(feature_names, dtype=object),
        )
        all_mc_rows.append({
            "seed": seed,
            "run_name": run_name,
            "formula_json": str(fjson),
            "top_k": args.top_k,
            "Z_shape": str(z_all.shape),
            "Z_mean": float(z_all.mean()),
            "model_checking_time_sec": float(mc_time),
            "mc_window": args.mc_window,
            "mc_tag": "fullcheck" if args.mc_window == 0 else f"{args.mc_truncation}{args.mc_window}check",
        })

        planned = []
        if planned_flags["verification_only"]:
            planned.append(("Verification-only LTL", "verification_only", "mlp", "verification_only", None))
        for L in args.raw_lengths:
            if planned_flags["action_ltl"]:
                planned.append(("Action + LTL", "action_ltl", "latefusion_action_ltl", str(L), L))
            if planned_flags["action_dwell_ltl"]:
                planned.append(("Action + Dwell + LTL", "action_dwell_ltl", "latefusion_action_dwell_ltl", str(L), L))
            if planned_flags["apc_raw"]:
                planned.append(("Action + Product + Category", "apc_raw", "transformer_apc", str(L), L))
            if planned_flags["apc_ltl"]:
                planned.append(("Action + Product + Category + LTL", "apc_ltl", "latefusion_apc_ltl", str(L), L))
            if planned_flags["adpc_raw"]:
                planned.append(("Action + Dwell + Product + Category", "adpc_raw", "transformer_adpc", str(L), L))
            if planned_flags["adpc_ltl"]:
                planned.append(("Action + Dwell + Product + Category + LTL", "adpc_ltl", "latefusion_adpc_ltl", str(L), L))
        if args.include_full_raw:
            L = full_len
            if planned_flags["apc_raw"]:
                planned.append(("Action + Product + Category", "apc_raw", "transformer_apc", "full", L))
            if planned_flags["adpc_raw"]:
                planned.append(("Action + Dwell + Product + Category", "adpc_raw", "transformer_adpc", "full", L))

        for config_name, experiment_type, model_name, length_label, raw_len in planned:
            print(f"\nRunning seed={seed} | {experiment_type} | length={length_label}")
            try:
                if experiment_type == "verification_only":
                    train_ds = base_exp.ZDataset(z_train, y_train)
                    val_ds = base_exp.ZDataset(z_val, y_val)
                    test_ds = base_exp.ZDataset(z_test, y_test)
                    model = base_exp.MLPBinaryClassifier(input_dim=z_train.shape[1], hidden_dim=args.mlp_hidden_dim, dropout=args.dropout)
                    mode = "z"
                    raw_seq_len_used = 0
                    dwell_used = False
                else:
                    a_train = base_exp.build_raw_matrix(train_df["_action_seq"].tolist(), action_vocab, raw_len, args.raw_truncation)
                    a_val = base_exp.build_raw_matrix(val_df["_action_seq"].tolist(), action_vocab, raw_len, args.raw_truncation)
                    a_test = base_exp.build_raw_matrix(test_df["_action_seq"].tolist(), action_vocab, raw_len, args.raw_truncation)

                    p_train = build_id_matrix(train_df["_product_seq"].tolist(), product_vocab, raw_len, args.raw_truncation)
                    p_val = build_id_matrix(val_df["_product_seq"].tolist(), product_vocab, raw_len, args.raw_truncation)
                    p_test = build_id_matrix(test_df["_product_seq"].tolist(), product_vocab, raw_len, args.raw_truncation)

                    c_train = build_id_matrix(train_df["_category_seq"].tolist(), category_vocab, raw_len, args.raw_truncation)
                    c_val = build_id_matrix(val_df["_category_seq"].tolist(), category_vocab, raw_len, args.raw_truncation)
                    c_test = build_id_matrix(test_df["_category_seq"].tolist(), category_vocab, raw_len, args.raw_truncation)

                    raw_seq_len_used = raw_len

                    if experiment_type == "action_ltl":
                        train_ds = base_exp.FusionDataset(a_train, z_train, y_train)
                        val_ds = base_exp.FusionDataset(a_val, z_val, y_val)
                        test_ds = base_exp.FusionDataset(a_test, z_test, y_test)
                        model = base_exp.LateFusionTransformerLTLClassifier(
                            vocab_size=len(action_vocab) + 1,
                            max_len=raw_len,
                            z_dim=z_train.shape[1],
                            d_model=args.d_model,
                            n_heads=args.n_heads,
                            n_layers=args.n_layers,
                            dim_feedforward=args.dim_feedforward,
                            ltl_hidden_dim=args.ltl_hidden_dim,
                            fusion_hidden_dim=args.fusion_hidden_dim,
                            dropout=args.dropout,
                        )
                        mode = "action_ltl"
                        dwell_used = False

                    elif experiment_type == "action_dwell_ltl":
                        mean, std = fit_dwell_scaler(train_df["_dwell_seq"].tolist(), raw_len, args.raw_truncation, log1p=True)
                        d_train = build_dwell_matrix(train_df["_dwell_seq"].tolist(), raw_len, args.raw_truncation, mean, std)
                        d_val = build_dwell_matrix(val_df["_dwell_seq"].tolist(), raw_len, args.raw_truncation, mean, std)
                        d_test = build_dwell_matrix(test_df["_dwell_seq"].tolist(), raw_len, args.raw_truncation, mean, std)
                        train_ds = base_exp.FusionDwellDataset(a_train, d_train, z_train, y_train)
                        val_ds = base_exp.FusionDwellDataset(a_val, d_val, z_val, y_val)
                        test_ds = base_exp.FusionDwellDataset(a_test, d_test, z_test, y_test)
                        model = base_exp.LateFusionDwellTransformerLTLClassifier(
                            vocab_size=len(action_vocab) + 1,
                            max_len=raw_len,
                            z_dim=z_train.shape[1],
                            d_model=args.d_model,
                            n_heads=args.n_heads,
                            n_layers=args.n_layers,
                            dim_feedforward=args.dim_feedforward,
                            ltl_hidden_dim=args.ltl_hidden_dim,
                            fusion_hidden_dim=args.fusion_hidden_dim,
                            dropout=args.dropout,
                        )
                        mode = "action_dwell_ltl"
                        dwell_used = True

                    elif experiment_type == "apc_raw":
                        train_ds = APCDataset(a_train, p_train, c_train, y_train)
                        val_ds = APCDataset(a_val, p_val, c_val, y_val)
                        test_ds = APCDataset(a_test, p_test, c_test, y_test)
                        model = APCTransformer(len(action_vocab) + 1, len(product_vocab), len(category_vocab), raw_len,
                                               use_dwell=False, d_model=args.d_model, n_heads=args.n_heads,
                                               n_layers=args.n_layers, dim_feedforward=args.dim_feedforward, dropout=args.dropout)
                        mode = "apc"
                        dwell_used = False

                    elif experiment_type == "apc_ltl":
                        train_ds = APCFusionDataset(a_train, p_train, c_train, z_train, y_train)
                        val_ds = APCFusionDataset(a_val, p_val, c_val, z_val, y_val)
                        test_ds = APCFusionDataset(a_test, p_test, c_test, z_test, y_test)
                        model = APCLateFusionTransformer(len(action_vocab) + 1, len(product_vocab), len(category_vocab), raw_len, z_train.shape[1],
                                                          use_dwell=False, d_model=args.d_model, n_heads=args.n_heads,
                                                          n_layers=args.n_layers, dim_feedforward=args.dim_feedforward,
                                                          ltl_hidden_dim=args.ltl_hidden_dim, fusion_hidden_dim=args.fusion_hidden_dim,
                                                          dropout=args.dropout)
                        mode = "apc_ltl"
                        dwell_used = False

                    elif experiment_type == "adpc_raw":
                        mean, std = fit_dwell_scaler(train_df["_dwell_seq"].tolist(), raw_len, args.raw_truncation, log1p=True)
                        d_train = build_dwell_matrix(train_df["_dwell_seq"].tolist(), raw_len, args.raw_truncation, mean, std)
                        d_val = build_dwell_matrix(val_df["_dwell_seq"].tolist(), raw_len, args.raw_truncation, mean, std)
                        d_test = build_dwell_matrix(test_df["_dwell_seq"].tolist(), raw_len, args.raw_truncation, mean, std)
                        train_ds = ADPCDataset(a_train, d_train, p_train, c_train, y_train)
                        val_ds = ADPCDataset(a_val, d_val, p_val, c_val, y_val)
                        test_ds = ADPCDataset(a_test, d_test, p_test, c_test, y_test)
                        model = APCTransformer(len(action_vocab) + 1, len(product_vocab), len(category_vocab), raw_len,
                                               use_dwell=True, d_model=args.d_model, n_heads=args.n_heads,
                                               n_layers=args.n_layers, dim_feedforward=args.dim_feedforward, dropout=args.dropout)
                        mode = "adpc"
                        dwell_used = True

                    elif experiment_type == "adpc_ltl":
                        mean, std = fit_dwell_scaler(train_df["_dwell_seq"].tolist(), raw_len, args.raw_truncation, log1p=True)
                        d_train = build_dwell_matrix(train_df["_dwell_seq"].tolist(), raw_len, args.raw_truncation, mean, std)
                        d_val = build_dwell_matrix(val_df["_dwell_seq"].tolist(), raw_len, args.raw_truncation, mean, std)
                        d_test = build_dwell_matrix(test_df["_dwell_seq"].tolist(), raw_len, args.raw_truncation, mean, std)
                        train_ds = ADPCFusionDataset(a_train, d_train, p_train, c_train, z_train, y_train)
                        val_ds = ADPCFusionDataset(a_val, d_val, p_val, c_val, z_val, y_val)
                        test_ds = ADPCFusionDataset(a_test, d_test, p_test, c_test, z_test, y_test)
                        model = APCLateFusionTransformer(len(action_vocab) + 1, len(product_vocab), len(category_vocab), raw_len, z_train.shape[1],
                                                          use_dwell=True, d_model=args.d_model, n_heads=args.n_heads,
                                                          n_layers=args.n_layers, dim_feedforward=args.dim_feedforward,
                                                          ltl_hidden_dim=args.ltl_hidden_dim, fusion_hidden_dim=args.fusion_hidden_dim,
                                                          dropout=args.dropout)
                        mode = "adpc_ltl"
                        dwell_used = True
                    else:
                        raise ValueError(experiment_type)

                train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers, pin_memory=True)
                val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, pin_memory=True)
                test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers, pin_memory=True)

                set_seed(seed)
                metrics = train_model(model, train_loader, val_loader, test_loader, y_train, mode)
                mc_used = 0.0 if experiment_type in {"apc_raw", "adpc_raw"} else float(mc_time)
                total_train_infer = metrics["train_time_sec"] + metrics["infer_time_sec"]
                total_pipeline = mc_used + total_train_infer
                row = {
                    "run_name": run_name,
                    "seed": seed,
                    "status": "ok",
                    "feature_pack": run_name,
                    "config_name": config_name,
                    "experiment_type": experiment_type,
                    "model": model_name,
                    "length_label": str(length_label),
                    "mc_tag": "fullcheck" if args.mc_window == 0 else f"{args.mc_truncation}{args.mc_window}check",
                    "top_k": args.top_k,
                    "formula_json": str(fjson),
                    "raw_sequence_length_used": int(raw_seq_len_used),
                    "dwell_sequence_used": bool(dwell_used),
                    "product_vocab_size": int(len(product_vocab)),
                    "category_vocab_size": int(len(category_vocab)),
                    "action_vocab_size": int(len(action_vocab)),
                    "model_checking_time_sec": float(mc_used),
                    "total_train_infer_time_sec": float(total_train_infer),
                    "total_pipeline_cost_sec": float(total_pipeline),
                    **metrics,
                }
                print(f"  f1={row['f1']:.4f} mcc={row['mcc']:.4f} auc={row['auc']:.4f} auprc={row['auprc']:.4f} brier={row['brier']:.4f} cost={row['total_pipeline_cost_sec']:.2f}s")
                all_rows.append(row)
            except Exception as e:
                print("  FAILED:", repr(e))
                all_rows.append({
                    "run_name": run_name,
                    "seed": seed,
                    "status": "failed",
                    "config_name": config_name,
                    "experiment_type": experiment_type,
                    "model": model_name,
                    "length_label": str(length_label),
                    "error": repr(e),
                })
            pd.DataFrame(all_rows).to_csv(out_dir / "all_runs.csv", index=False)

    pd.DataFrame(all_mc_rows).to_csv(out_dir / "model_checking_by_seed.csv", index=False)
    runs_df = pd.DataFrame(all_rows)
    ok = runs_df[runs_df["status"] == "ok"].copy()
    group_cols = ["run_name", "config_name", "experiment_type", "model", "length_label", "mc_tag"]
    metric_cols = [
        "f1", "mcc", "auc", "auprc", "brier", "logloss", "ece", "balanced_accuracy", "precision", "recall",
        "train_time_sec", "infer_time_sec", "model_checking_time_sec", "total_pipeline_cost_sec",
        "epochs_ran", "best_epoch", "n_params", "product_vocab_size", "category_vocab_size",
    ]
    summary_rows = []
    for keys, g in ok.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_runs"] = len(g)
        for col in metric_cols:
            if col in g.columns:
                row[f"{col}_mean"] = pd.to_numeric(g[col], errors="coerce").mean()
                row[f"{col}_std"] = pd.to_numeric(g[col], errors="coerce").std(ddof=1)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    keep_cols = [
        "run_name", "length_label", "config_name", "n_runs",
        "f1_mean", "mcc_mean", "auc_mean", "auprc_mean", "brier_mean",
        "total_pipeline_cost_sec_mean", "epochs_ran_mean",
    ]
    paper_df = summary_df[[c for c in keep_cols if c in summary_df.columns]].copy()
    if not paper_df.empty:
        order = {str(x): i for i, x in enumerate(args.raw_lengths)}
        order["full"] = 999
        order["verification_only"] = -1
        paper_df["_len_order"] = paper_df["length_label"].map(order).fillna(500)
        paper_df = paper_df.sort_values(["_len_order", "config_name"]).drop(columns=["_len_order"])
        paper_df.to_csv(out_dir / "paper_table.csv", index=False)

    print("\nDone.")
    print("Output:", out_dir)
    print("all_runs:", out_dir / "all_runs.csv")
    print("summary:", out_dir / "summary.csv")
    print("paper_table:", out_dir / "paper_table.csv")


if __name__ == "__main__":
    main()
