#!/usr/bin/env python3
"""
electronics_ltlf_pipeline.py

Unified pipeline for the Kaggle electronics clickstream dataset:
  1) Convert event-level logs to session-level traces compatible with the prior LTLf experiments.
  2) Create aligned train/val/test splits with random negative sampling.
  3) Model-check existing LTLf formula JSON files into fixed-dimensional Z matrices.
  4) Train raw, verification-only, and late-fusion Transformer/MLP baselines.

The script is intentionally parameterized so you can reuse it for:
  - behavior-only experiments
  - behavior+dwell experiments
  - behavior+product/category raw baselines
  - raw+LTL late fusion experiments

Important experimental convention:
  Purchase events define the label but are excluded from the input trace to avoid leakage.
  For positive sessions, the input trace is truncated before the first purchase event.

Example commands are provided at the bottom of this file and in the ChatGPT response.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, brier_score_loss
except Exception as e:
    raise ImportError("Please install scikit-learn: pip install scikit-learn") from e

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except Exception:
    torch = None
    nn = None
    Dataset = object
    DataLoader = None


# ============================================================
# General utilities
# ============================================================

ACTION_TO_ID = {"PAD": 0, "b": 1, "d": 2, "a": 3, "r": 4}
ID_TO_ACTION = {v: k for k, v in ACTION_TO_ID.items()}
DWELL_BUCKET_TO_ID = {"PAD": 0, "dl": 1, "dm": 2, "dh": 3}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def safe_load_list(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return []
        try:
            return json.loads(x)
        except Exception:
            return ast.literal_eval(x)
    try:
        return list(x)
    except Exception:
        return []


def dump_list(x: Sequence[Any]) -> str:
    return json.dumps(list(x), ensure_ascii=False)


def stringify_id(x: Any) -> str:
    if pd.isna(x):
        return "__NA__"
    return str(x)


def read_input_events(input_path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """Read one CSV or concatenate all CSV files in a directory."""
    if input_path.is_dir():
        csv_files = sorted(input_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {input_path}")
        dfs = []
        remaining = nrows
        for fp in csv_files:
            this_nrows = None if remaining is None else max(0, remaining)
            if remaining is not None and remaining <= 0:
                break
            print(f"Reading {fp} ...")
            df_part = pd.read_csv(
                fp,
                nrows=this_nrows,
                dtype={
                    "event_type": "string",
                    "product_id": "string",
                    "category_id": "string",
                    "category_code": "string",
                    "brand": "string",
                    "user_id": "string",
                    "user_session": "string",
                },
            )
            dfs.append(df_part)
            if remaining is not None:
                remaining -= len(df_part)
        return pd.concat(dfs, ignore_index=True)

    if not input_path.exists():
        raise FileNotFoundError(input_path)
    print(f"Reading {input_path} ...")
    return pd.read_csv(
        input_path,
        nrows=nrows,
        dtype={
            "event_type": "string",
            "product_id": "string",
            "category_id": "string",
            "category_code": "string",
            "brand": "string",
            "user_id": "string",
            "user_session": "string",
        },
    )


def factorize_sequence_column(df: pd.DataFrame, col: str, out_col: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    vals: List[str] = []
    for seq in df[col].map(safe_load_list):
        vals.extend([stringify_id(v) for v in seq])
    uniq = sorted(set(vals))
    mapping = {v: i + 1 for i, v in enumerate(uniq)}  # 0 reserved for padding
    encoded = []
    for seq in df[col].map(safe_load_list):
        encoded.append([mapping.get(stringify_id(v), 0) for v in seq])
    df[out_col] = [dump_list(x) for x in encoded]
    return df, mapping


# ============================================================
# Step 1. Prepare session-level traces
# ============================================================

def prepare_sessions(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_input_events(input_path, nrows=args.nrows)

    required = {"event_time", "event_type", "product_id", "category_id", "user_id", "user_session"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")

    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["event_time", "event_type", "user_session"]).copy()
    df["event_type"] = df["event_type"].astype(str).str.lower().str.strip()
    df = df.drop_duplicates()

    observed_types = sorted(df["event_type"].dropna().unique().tolist())
    print("Observed event_type values:", observed_types)

    # Optional session split by inactivity gap. Kaggle already has user_session, so default is None.
    df = df.sort_values(["user_session", "event_time"]).copy()
    if args.max_gap_hours is not None:
        gap_sec = df.groupby("user_session")["event_time"].diff().dt.total_seconds()
        new_block = (gap_sec.isna() | (gap_sec > args.max_gap_hours * 3600)).astype(int)
        df["_session_block"] = new_block.groupby(df["user_session"]).cumsum()
        df["session_id_hash"] = df["user_session"].astype(str) + "__" + df["_session_block"].astype(str)
    else:
        df["session_id_hash"] = df["user_session"].astype(str)

    # Compute timestamp-derived dwell proxy.
    df = df.sort_values(["session_id_hash", "event_time"]).copy()
    df["_next_time"] = df.groupby("session_id_hash")["event_time"].shift(-1)
    dwell = (df["_next_time"] - df["event_time"]).dt.total_seconds()
    dwell = dwell.fillna(0).clip(lower=0)
    if args.max_dwell_sec is not None:
        dwell = dwell.clip(upper=float(args.max_dwell_sec))
    df["_dwell_sec"] = dwell

    # Map event vocabulary to the old behavioral atoms.
    # Purchase is label-only and excluded from the input sequence.
    if args.view_as not in {"b", "d"}:
        raise ValueError("--view_as must be b or d")
    action_map = {
        "view": args.view_as,
        "cart": "a",
        "remove_from_cart": "r",
        "purchase": "p",
    }
    df["_action"] = df["event_type"].map(action_map).fillna("b")

    rows: List[Dict[str, Any]] = []
    for sid, g in df.groupby("session_id_hash", sort=False):
        g = g.sort_values("event_time").copy()
        events = g["event_type"].astype(str).tolist()
        purchase = int("purchase" in set(events))

        if purchase:
            first_purchase_pos = events.index("purchase")
            input_g = g.iloc[:first_purchase_pos].copy()
        else:
            input_g = g.copy()

        # Exclude purchase event from input to avoid leakage.
        input_g = input_g[input_g["event_type"] != "purchase"].copy()
        if len(input_g) < args.min_input_len:
            continue

        actions = input_g["_action"].astype(str).tolist()
        products = input_g["product_id"].map(stringify_id).tolist()
        cats = input_g["category_id"].map(stringify_id).tolist()
        if "category_code" in input_g.columns:
            cat_codes = input_g["category_code"].map(stringify_id).tolist()
        else:
            cat_codes = ["__NA__"] * len(input_g)
        if "brand" in input_g.columns:
            brands = input_g["brand"].map(stringify_id).tolist()
        else:
            brands = ["__NA__"] * len(input_g)
        if "price" in input_g.columns:
            prices = pd.to_numeric(input_g["price"], errors="coerce").fillna(0.0).astype(float).tolist()
        else:
            prices = [0.0] * len(input_g)
        dwell_seq = input_g["_dwell_sec"].astype(float).tolist()

        rows.append({
            "session_id_hash": sid,
            "user_id": stringify_id(input_g["user_id"].iloc[0]),
            "behavior_sequence": dump_list(actions),
            "action_sequence": dump_list(actions),
            "product_id_sequence": dump_list(products),
            "category_id_sequence": dump_list(cats),
            "category_code_sequence": dump_list(cat_codes),
            "brand_sequence": dump_list(brands),
            "price_sequence": dump_list(prices),
            "dwell_time_sequence": dump_list(dwell_seq),
            "purchase": purchase,
            "session_len": len(actions),
            "n_distinct_products": len(set(products)),
            "n_distinct_categories": len(set(cats)),
            "start_time": input_g["event_time"].iloc[0].isoformat(),
            "end_time": input_g["event_time"].iloc[-1].isoformat(),
        })

    sess = pd.DataFrame(rows)
    if sess.empty:
        raise RuntimeError("No valid sessions after preprocessing. Try --min_input_len 1.")

    # Optional debugging sample after session construction.
    if args.max_sessions is not None and len(sess) > args.max_sessions:
        sess, _ = train_test_split(
            sess,
            train_size=args.max_sessions,
            stratify=sess["purchase"],
            random_state=args.sample_seed,
        )
        sess = sess.reset_index(drop=True)

    # Build dwell buckets. Quantile buckets are dataset-specific; keep this in Methods.
    all_dwell = []
    for seq in sess["dwell_time_sequence"].map(safe_load_list):
        all_dwell.extend([float(x) for x in seq if float(x) > 0])
    if args.dwell_q1 is not None and args.dwell_q2 is not None:
        q1, q2 = float(args.dwell_q1), float(args.dwell_q2)
    elif len(all_dwell) > 0:
        q1, q2 = np.quantile(all_dwell, [1 / 3, 2 / 3]).tolist()
    else:
        q1, q2 = 1.0, 5.0
    print(f"Dwell bucket thresholds: q1={q1:.3f}s, q2={q2:.3f}s")

    def bucketize_dwell(seq: Any) -> List[str]:
        out = []
        for v in safe_load_list(seq):
            x = float(v)
            if x <= q1:
                out.append("dl")
            elif x <= q2:
                out.append("dm")
            else:
                out.append("dh")
        return out

    sess["dwell_bucket_sequence"] = [dump_list(bucketize_dwell(x)) for x in sess["dwell_time_sequence"]]

    # Build atom_sequence containing behavior, dwell, product relation, and category relation atoms.
    # Behavior-only formulas simply ignore extra atoms.
    atom_sequences = []
    for _, row in sess.iterrows():
        actions = safe_load_list(row["behavior_sequence"])
        products = safe_load_list(row["product_id_sequence"])
        cats = safe_load_list(row["category_id_sequence"])
        dwell_buckets = safe_load_list(row["dwell_bucket_sequence"])

        seen_products: Set[str] = set()
        seen_cats: Set[str] = set()
        steps: List[List[str]] = []
        for i, a in enumerate(actions):
            p = stringify_id(products[i]) if i < len(products) else "__NA__"
            c = stringify_id(cats[i]) if i < len(cats) else "__NA__"
            atoms: Set[str] = set()

            # Behavioral atom.
            atoms.add(str(a))

            # Dwell atom.
            if i < len(dwell_buckets):
                atoms.add(str(dwell_buckets[i]))

            # Product-relation atoms.
            if p not in seen_products:
                atoms.add("fp")  # first exposure of this product within session
            else:
                atoms.add("rp")  # repeated/revisited product within session
            if i > 0:
                prev_p = stringify_id(products[i - 1])
                atoms.add("sp" if p == prev_p else "pc")
            distinct_products_so_far = len(seen_products | {p})
            if distinct_products_so_far >= 2:
                atoms.add("p2")
            if distinct_products_so_far >= 3:
                atoms.add("p3")

            # Category-relation atoms.
            if i > 0:
                prev_c = stringify_id(cats[i - 1])
                atoms.add("sc" if c == prev_c else "cc")
            if c in seen_cats:
                atoms.add("rc")

            seen_products.add(p)
            seen_cats.add(c)
            steps.append(sorted(atoms))
        atom_sequences.append(dump_list(steps))
    sess["atom_sequence"] = atom_sequences

    # Raw branch opaque categorical IDs.
    sess, product_map = factorize_sequence_column(sess, "product_id_sequence", "product_sku_hash_sequence")
    sess, category_map = factorize_sequence_column(sess, "category_id_sequence", "category_hash_sequence")

    # Save outputs.
    sessions_csv = out_dir / "electronics_sessions_ltl_ready.csv"
    sess.to_csv(sessions_csv, index=False)
    with open(out_dir / "product_id_mapping.json", "w") as f:
        json.dump(product_map, f)
    with open(out_dir / "category_id_mapping.json", "w") as f:
        json.dump(category_map, f)

    summary = {
        "n_sessions": int(len(sess)),
        "n_positive": int(sess["purchase"].sum()),
        "n_negative": int((1 - sess["purchase"]).sum()),
        "positive_rate": float(sess["purchase"].mean()),
        "session_len_mean": float(sess["session_len"].mean()),
        "session_len_median": float(sess["session_len"].median()),
        "session_len_p95": float(sess["session_len"].quantile(0.95)),
        "session_len_max": int(sess["session_len"].max()),
        "n_product_ids": int(len(product_map)),
        "n_category_ids": int(len(category_map)),
        "view_as": args.view_as,
        "max_dwell_sec": args.max_dwell_sec,
        "dwell_q1_sec": float(q1),
        "dwell_q2_sec": float(q2),
        "purchase_event_excluded_from_input": True,
        "positive_sessions_truncated_before_first_purchase": True,
    }
    with open(out_dir / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved sessions:", sessions_csv)
    print(json.dumps(summary, indent=2))

    # Create splits immediately.
    make_splits_from_sessions(sess, out_dir, args.seeds, args.neg_ratio, args.train_size, args.val_size, args.test_size)


def make_splits_from_sessions(
    sess: pd.DataFrame,
    out_dir: Path,
    seeds: Sequence[int],
    neg_ratio: int,
    train_size: float,
    val_size: float,
    test_size: float,
) -> None:
    assert abs(train_size + val_size + test_size - 1.0) < 1e-8
    split_root = out_dir / "electronics_splits"
    split_root.mkdir(parents=True, exist_ok=True)

    pos = sess[sess["purchase"] == 1].copy()
    neg = sess[sess["purchase"] == 0].copy()
    if len(pos) == 0 or len(neg) == 0:
        raise RuntimeError("Need both positive and negative sessions for stratified splits.")

    print(f"Creating splits: all={len(sess)} pos={len(pos)} neg={len(neg)} neg_ratio={neg_ratio}")
    for seed in seeds:
        n_neg = min(len(neg), len(pos) * neg_ratio)
        neg_sample = neg.sample(n=n_neg, random_state=seed)
        data = pd.concat([pos, neg_sample], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)

        train_df, temp_df = train_test_split(
            data,
            train_size=train_size,
            stratify=data["purchase"],
            random_state=seed,
        )
        rel_val = val_size / (val_size + test_size)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=rel_val,
            stratify=temp_df["purchase"],
            random_state=seed,
        )

        seed_dir = split_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(seed_dir / "train.csv", index=False)
        val_df.to_csv(seed_dir / "val.csv", index=False)
        test_df.to_csv(seed_dir / "test.csv", index=False)

        print(
            f"seed={seed} | train={len(train_df)} val={len(val_df)} test={len(test_df)} "
            f"| pos(train/val/test)={int(train_df.purchase.sum())}/{int(val_df.purchase.sum())}/{int(test_df.purchase.sum())}"
        )


# ============================================================
# Step 2. LTLf model checking for existing formula JSON
# ============================================================

@dataclass(frozen=True)
class LTLNode:
    op: str
    left: Any = None
    right: Any = None


TOKEN_RE = re.compile(r"\s*(\(|\)|&&|\|\||!|U|X|true|false|[A-Za-z_][A-Za-z0-9_]*)\s*")


def tokenize_formula(s: str) -> List[str]:
    toks = TOKEN_RE.findall(s)
    if not toks:
        raise ValueError(f"Could not tokenize formula: {s}")
    return toks


class FormulaParser:
    def __init__(self, toks: List[str]):
        self.toks = toks
        self.i = 0

    def peek(self) -> Optional[str]:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def pop(self, expected: Optional[str] = None) -> str:
        if self.i >= len(self.toks):
            raise ValueError("Unexpected end of formula")
        tok = self.toks[self.i]
        if expected is not None and tok != expected:
            raise ValueError(f"Expected {expected}, got {tok}; toks={self.toks}")
        self.i += 1
        return tok

    def parse(self) -> LTLNode:
        node = self.parse_or()
        if self.i != len(self.toks):
            raise ValueError(f"Unused tokens: {self.toks[self.i:]}")
        return node

    def parse_or(self) -> LTLNode:
        node = self.parse_and()
        while self.peek() == "||":
            self.pop("||")
            node = LTLNode("or", node, self.parse_and())
        return node

    def parse_and(self) -> LTLNode:
        node = self.parse_until()
        while self.peek() == "&&":
            self.pop("&&")
            node = LTLNode("and", node, self.parse_until())
        return node

    def parse_until(self) -> LTLNode:
        node = self.parse_unary()
        while self.peek() == "U":
            self.pop("U")
            node = LTLNode("U", node, self.parse_unary())
        return node

    def parse_unary(self) -> LTLNode:
        tok = self.peek()
        if tok == "!":
            self.pop("!")
            return LTLNode("not", self.parse_unary())
        if tok == "X":
            self.pop("X")
            return LTLNode("X", self.parse_unary())
        if tok == "(":
            self.pop("(")
            node = self.parse_or()
            self.pop(")")
            return node
        tok = self.pop()
        if tok == "true":
            return LTLNode("true")
        if tok == "false":
            return LTLNode("false")
        return LTLNode("atom", tok)


def parse_formula(s: str) -> LTLNode:
    return FormulaParser(tokenize_formula(s)).parse()


def eval_ltl(node: LTLNode, trace: List[Set[str]], t: int, memo: Dict[Tuple[LTLNode, int], bool]) -> bool:
    key = (node, t)
    if key in memo:
        return memo[key]
    n = len(trace)
    if node.op == "true":
        ans = True
    elif node.op == "false":
        ans = False
    elif node.op == "atom":
        ans = (t < n) and (str(node.left) in trace[t])
    elif node.op == "not":
        ans = not eval_ltl(node.left, trace, t, memo)
    elif node.op == "and":
        ans = eval_ltl(node.left, trace, t, memo) and eval_ltl(node.right, trace, t, memo)
    elif node.op == "or":
        ans = eval_ltl(node.left, trace, t, memo) or eval_ltl(node.right, trace, t, memo)
    elif node.op == "X":
        ans = (t + 1 < n) and eval_ltl(node.left, trace, t + 1, memo)
    elif node.op == "U":
        ans = False
        prefix_ok = True
        for j in range(t, n):
            if prefix_ok and eval_ltl(node.right, trace, j, memo):
                ans = True
                break
            if not eval_ltl(node.left, trace, j, memo):
                prefix_ok = False
                break
    else:
        raise ValueError(f"Unknown op: {node.op}")
    memo[key] = ans
    return ans


def load_formula_strings(formula_json: Path, top_k: Optional[int] = None) -> Tuple[List[str], List[Dict[str, Any]]]:
    with open(formula_json, "r") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        rows = obj
    elif isinstance(obj, dict):
        rows = None
        for key in ["formulas", "selected_formulas", "top_formulas", "data"]:
            if key in obj and isinstance(obj[key], list):
                rows = obj[key]
                break
        if rows is None:
            raise ValueError(f"Cannot find formula list in {formula_json}")
    else:
        raise ValueError(f"Unsupported formula JSON format: {formula_json}")

    formulas: List[str] = []
    meta_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        if isinstance(row, str):
            formulas.append(row)
            meta_rows.append({"rank": i + 1, "formula_str": row})
        elif isinstance(row, dict):
            f = None
            for k in ["formula_str", "formula", "ltlf", "expr"]:
                if k in row:
                    f = str(row[k])
                    break
            if f is None:
                continue
            formulas.append(f)
            m = dict(row)
            m.setdefault("rank", i + 1)
            meta_rows.append(m)
    if top_k is not None:
        formulas = formulas[:top_k]
        meta_rows = meta_rows[:top_k]
    if not formulas:
        raise ValueError(f"No formulas loaded from {formula_json}")
    return formulas, meta_rows


def atom_sequence_to_trace(x: Any) -> List[Set[str]]:
    steps = safe_load_list(x)
    trace: List[Set[str]] = []
    for step in steps:
        if isinstance(step, list):
            trace.append(set(map(str, step)))
        elif isinstance(step, str):
            trace.append(set([s for s in re.split(r"[,\s]+", step.strip()) if s]))
        else:
            try:
                trace.append(set(map(str, list(step))))
            except Exception:
                trace.append(set())
    return trace


def cmd_modelcheck(args: argparse.Namespace) -> None:
    sessions_csv = Path(args.sessions_csv)
    formula_json = Path(args.formula_json)
    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(sessions_csv)
    formulas, meta = load_formula_strings(formula_json, top_k=args.top_k)
    print(f"Loaded {len(formulas)} formulas from {formula_json}")
    nodes = [parse_formula(f) for f in formulas]

    Z = np.zeros((len(df), len(nodes)), dtype=np.uint8)
    t0 = time.time()
    for i, x in enumerate(df[args.atom_col].tolist()):
        trace = atom_sequence_to_trace(x)
        for j, node in enumerate(nodes):
            Z[i, j] = int(eval_ltl(node, trace, 0, memo={}))
        if args.progress_every > 0 and (i + 1) % args.progress_every == 0:
            print(f"model checked {i + 1}/{len(df)} traces")
    elapsed = time.time() - t0
    print(f"Model checking finished: Z={Z.shape}, time={elapsed:.2f}s")

    y = df["purchase"].astype(int).to_numpy()
    sid = df["session_id_hash"].astype(str).to_numpy()
    np.savez_compressed(out_npz, Z=Z, y=y, session_id_hash=sid, formulas=np.array(formulas, dtype=object))

    # Save diagnostic support rates.
    stats = []
    for j, f in enumerate(formulas):
        z = Z[:, j]
        pos_mask = y == 1
        neg_mask = y == 0
        stats.append({
            "rank": j + 1,
            "formula_str": f,
            "activation_rate_all": float(z.mean()),
            "activation_rate_pos": float(z[pos_mask].mean()) if pos_mask.any() else np.nan,
            "activation_rate_neg": float(z[neg_mask].mean()) if neg_mask.any() else np.nan,
            "support_gap": float(z[pos_mask].mean() - z[neg_mask].mean()) if pos_mask.any() and neg_mask.any() else np.nan,
        })
    stats_csv = out_npz.with_suffix(".formula_support.csv")
    pd.DataFrame(stats).to_csv(stats_csv, index=False)
    meta_json = out_npz.with_suffix(".meta.json")
    with open(meta_json, "w") as f:
        json.dump({
            "formula_json": str(formula_json),
            "top_k": args.top_k,
            "atom_col": args.atom_col,
            "n_sessions": int(len(df)),
            "model_checking_time_sec": elapsed,
            "out_npz": str(out_npz),
            "support_csv": str(stats_csv),
        }, f, indent=2)
    print("Saved:", out_npz)
    print("Saved diagnostics:", stats_csv)


# ============================================================
# Step 3. Dataset and models for experiments
# ============================================================

class ClickDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_pack: str,
        max_len: int,
        z_map: Optional[Dict[str, np.ndarray]] = None,
        dwell_scale: Optional[float] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.feature_pack = feature_pack
        self.max_len = max_len
        self.z_map = z_map
        self.dwell_scale = dwell_scale if dwell_scale is not None else 1.0
        self.use_z = feature_pack in {"verification_only", "b_ltl", "bd_ltl", "apc_ltl", "adpc_ltl"}
        self.use_product = feature_pack in {"apc_raw", "adpc_raw", "apc_ltl", "adpc_ltl"}
        self.use_dwell = feature_pack in {"bd_raw", "adpc_raw", "bd_ltl", "adpc_ltl"}
        if self.use_z and z_map is None:
            raise ValueError(f"feature_pack={feature_pack} requires --z_npz")

    def __len__(self) -> int:
        return len(self.df)

    def _last_pad(self, values: List[Any], pad_value: Any, dtype=np.int64) -> Tuple[np.ndarray, np.ndarray]:
        values = list(values)[-self.max_len:]
        n = len(values)
        arr = np.full(self.max_len, pad_value, dtype=dtype)
        mask = np.zeros(self.max_len, dtype=np.float32)
        if n > 0:
            arr[-n:] = np.asarray(values, dtype=dtype)
            mask[-n:] = 1.0
        return arr, mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        sid = str(row["session_id_hash"])
        actions = [ACTION_TO_ID.get(str(a), 0) for a in safe_load_list(row["behavior_sequence"])]
        action_arr, mask = self._last_pad(actions, 0, dtype=np.int64)

        out = {
            "sid": sid,
            "action": torch.tensor(action_arr, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "y": torch.tensor(float(row["purchase"]), dtype=torch.float32),
        }

        if self.use_product:
            prod = [int(x) for x in safe_load_list(row["product_sku_hash_sequence"])]
            cat = [int(x) for x in safe_load_list(row["category_hash_sequence"])]
            prod_arr, _ = self._last_pad(prod, 0, dtype=np.int64)
            cat_arr, _ = self._last_pad(cat, 0, dtype=np.int64)
            out["product"] = torch.tensor(prod_arr, dtype=torch.long)
            out["category"] = torch.tensor(cat_arr, dtype=torch.long)

        if self.use_dwell:
            dwell = [math.log1p(float(x)) / self.dwell_scale for x in safe_load_list(row["dwell_time_sequence"])]
            dwell_arr, _ = self._last_pad(dwell, 0.0, dtype=np.float32)
            out["dwell"] = torch.tensor(dwell_arr, dtype=torch.float32)

        if self.use_z:
            z = self.z_map[sid]
            out["z"] = torch.tensor(z.astype(np.float32), dtype=torch.float32)
        return out


class RawEncoder(nn.Module):
    def __init__(
        self,
        n_actions: int,
        n_products: int,
        n_categories: int,
        feature_pack: str,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.feature_pack = feature_pack
        self.use_product = feature_pack in {"apc_raw", "adpc_raw", "apc_ltl", "adpc_ltl"}
        self.use_dwell = feature_pack in {"bd_raw", "adpc_raw", "bd_ltl", "adpc_ltl"}

        self.action_emb = nn.Embedding(n_actions, d_model, padding_idx=0)
        if self.use_product:
            self.product_emb = nn.Embedding(n_products, d_model, padding_idx=0)
            self.category_emb = nn.Embedding(n_categories, d_model, padding_idx=0)
        if self.use_dwell:
            self.dwell_proj = nn.Linear(1, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.action_emb(batch["action"])
        if self.use_product:
            x = x + self.product_emb(batch["product"]) + self.category_emb(batch["category"])
        if self.use_dwell:
            x = x + self.dwell_proj(batch["dwell"].unsqueeze(-1))
        B, L, D = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        x = x + self.pos_emb(pos)
        pad_mask = batch["mask"] <= 0
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        mask = batch["mask"].unsqueeze(-1)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.norm(pooled)


class FusionClassifier(nn.Module):
    def __init__(
        self,
        feature_pack: str,
        n_products: int,
        n_categories: int,
        z_dim: int = 0,
        d_model: int = 64,
        z_hidden: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.feature_pack = feature_pack
        self.is_verification_only = feature_pack == "verification_only"
        self.use_z = feature_pack in {"verification_only", "b_ltl", "bd_ltl", "apc_ltl", "adpc_ltl"}
        self.use_raw = feature_pack in {"b_raw", "bd_raw", "apc_raw", "adpc_raw", "b_ltl", "bd_ltl", "apc_ltl", "adpc_ltl"}

        if self.use_raw:
            self.raw_encoder = RawEncoder(
                n_actions=len(ACTION_TO_ID),
                n_products=n_products,
                n_categories=n_categories,
                feature_pack=feature_pack,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout,
                max_len=max_len,
            )
        if self.use_z:
            if z_dim <= 0:
                raise ValueError("z_dim must be >0 for LTL feature packs")
            self.z_encoder = nn.Sequential(
                nn.Linear(z_dim, z_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(z_hidden, z_hidden),
                nn.ReLU(),
            )

        in_dim = 0
        if self.use_raw:
            in_dim += d_model
        if self.use_z:
            in_dim += z_hidden
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        if self.use_raw:
            parts.append(self.raw_encoder(batch))
        if self.use_z:
            parts.append(self.z_encoder(batch["z"]))
        h = torch.cat(parts, dim=-1)
        return self.classifier(h).squeeze(-1)


def load_z_map(z_npz: Optional[str]) -> Tuple[Optional[Dict[str, np.ndarray]], int]:
    if z_npz is None:
        return None, 0
    z = np.load(z_npz, allow_pickle=True)
    Z = z["Z"]
    sids = z["session_id_hash"].astype(str)
    z_map = {sid: Z[i].astype(np.float32) for i, sid in enumerate(sids)}
    return z_map, Z.shape[1]


def get_vocab_sizes(sessions_csv: Path) -> Tuple[int, int]:
    df = pd.read_csv(sessions_csv, usecols=["product_sku_hash_sequence", "category_hash_sequence"])
    max_prod = 0
    max_cat = 0
    for seq in df["product_sku_hash_sequence"].map(safe_load_list):
        if seq:
            max_prod = max(max_prod, max(map(int, seq)))
    for seq in df["category_hash_sequence"].map(safe_load_list):
        if seq:
            max_cat = max(max_cat, max(map(int, seq)))
    return max_prod + 1, max_cat + 1


def determine_max_len(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, length: str, cap: Optional[int]) -> int:
    if length.lower() != "full":
        return int(length)
    max_len = int(max(train_df["session_len"].max(), val_df["session_len"].max(), test_df["session_len"].max()))
    if cap is not None:
        max_len = min(max_len, int(cap))
    return max_len


def compute_dwell_scale(train_df: pd.DataFrame) -> float:
    vals = []
    for seq in train_df["dwell_time_sequence"].map(safe_load_list):
        vals.extend([math.log1p(float(x)) for x in seq if float(x) > 0])
    if not vals:
        return 1.0
    return float(np.percentile(vals, 95) or 1.0)


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    probs: List[float] = []
    ys: List[int] = []
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            logits = model(batch)
            p = torch.sigmoid(logits).detach().cpu().numpy()
            probs.extend(p.tolist())
            ys.extend(batch["y"].detach().cpu().numpy().astype(int).tolist())
    infer_time = time.time() - t0
    y = np.asarray(ys).astype(int)
    p = np.asarray(probs)
    pred = (p >= 0.5).astype(int)
    out = {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(y)) > 1 else np.nan,
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "infer_time_sec": float(infer_time),
    }
    return out


def train_one_seed(args: argparse.Namespace, seed: int) -> Dict[str, Any]:
    if torch is None:
        raise ImportError("PyTorch is required for train mode.")
    set_seed(seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    splits_dir = Path(args.splits_dir)
    seed_dir = splits_dir / f"seed_{seed}"
    train_df = pd.read_csv(seed_dir / "train.csv")
    val_df = pd.read_csv(seed_dir / "val.csv")
    test_df = pd.read_csv(seed_dir / "test.csv")
    max_len = determine_max_len(train_df, val_df, test_df, args.length, args.full_max_len)
    print(f"seed={seed} feature_pack={args.feature_pack} length={args.length} -> max_len={max_len}")

    z_map, z_dim = load_z_map(args.z_npz)
    if z_map is not None:
        # Basic alignment check.
        missing = [sid for sid in pd.concat([train_df, val_df, test_df])["session_id_hash"].astype(str).tolist() if sid not in z_map]
        if missing:
            raise ValueError(f"Z matrix missing {len(missing)} session IDs. Example: {missing[:3]}")

    n_products, n_categories = get_vocab_sizes(Path(args.sessions_csv))
    dwell_scale = compute_dwell_scale(train_df)

    train_ds = ClickDataset(train_df, args.feature_pack, max_len, z_map=z_map, dwell_scale=dwell_scale)
    val_ds = ClickDataset(val_df, args.feature_pack, max_len, z_map=z_map, dwell_scale=dwell_scale)
    test_ds = ClickDataset(test_df, args.feature_pack, max_len, z_map=z_map, dwell_scale=dwell_scale)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = FusionClassifier(
        feature_pack=args.feature_pack,
        n_products=n_products,
        n_categories=n_categories,
        z_dim=z_dim,
        d_model=args.d_model,
        z_hidden=args.z_hidden,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_len=max_len,
    ).to(device)

    # Class imbalance handling.
    pos = float(train_df["purchase"].sum())
    neg = float(len(train_df) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = -np.inf
    best_state = None
    patience_left = args.patience
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, batch["y"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        val_metrics = evaluate_model(model, val_loader, device)
        monitor = val_metrics[args.monitor]
        print(f"epoch={epoch:03d} loss={np.mean(losses):.4f} val_{args.monitor}={monitor:.4f} val_f1={val_metrics['f1']:.4f} val_auc={val_metrics['auc']:.4f}")
        if monitor > best_val:
            best_val = monitor
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break
    train_time = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_model(model, test_loader, device)
    result = {
        "seed": seed,
        "dataset": args.dataset_name,
        "feature_pack": args.feature_pack,
        "model": "transformer_late_fusion" if "ltl" in args.feature_pack else ("mlp" if args.feature_pack == "verification_only" else "transformer"),
        "length_label": args.length,
        "max_len_used": max_len,
        "z_dim": z_dim,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "train_time_sec": float(train_time),
        **test_metrics,
    }
    print("TEST", result)
    return result


def cmd_train(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        rows.append(train_one_seed(args, seed))
        # Save incrementally.
        pd.DataFrame(rows).to_csv(out_dir / "all_runs.csv", index=False)
    df = pd.DataFrame(rows)
    metric_cols = ["f1", "mcc", "auc", "auprc", "brier", "train_time_sec", "infer_time_sec"]
    summary = df.groupby(["dataset", "feature_pack", "model", "length_label", "z_dim"], dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    # Flatten columns.
    summary.columns = ["_".join([str(c) for c in col if c != ""]).strip("_") for col in summary.columns.values]
    summary.to_csv(out_dir / "summary_mean_std.csv", index=False)
    print("Saved:", out_dir / "all_runs.csv")
    print("Saved:", out_dir / "summary_mean_std.csv")


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Kaggle electronics LTLf clickstream pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    # prepare
    sp = sub.add_parser("prepare", help="Convert event-level Kaggle data to session-level traces and splits")
    sp.add_argument("--input", required=True, help="CSV file or directory containing CSV files")
    sp.add_argument("--out_dir", required=True)
    sp.add_argument("--view_as", choices=["b", "d"], default="d", help="Map Kaggle view to old atom b or d. Default d.")
    sp.add_argument("--min_input_len", type=int, default=2)
    sp.add_argument("--max_gap_hours", type=float, default=None)
    sp.add_argument("--max_dwell_sec", type=float, default=1800.0, help="Clip timestamp-derived dwell proxy. Use -1 to disable.")
    sp.add_argument("--dwell_q1", type=float, default=None)
    sp.add_argument("--dwell_q2", type=float, default=None)
    sp.add_argument("--neg_ratio", type=int, default=4)
    sp.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    sp.add_argument("--train_size", type=float, default=0.70)
    sp.add_argument("--val_size", type=float, default=0.10)
    sp.add_argument("--test_size", type=float, default=0.20)
    sp.add_argument("--nrows", type=int, default=None, help="Optional for debugging")
    sp.add_argument("--max_sessions", type=int, default=None, help="Optional stratified session sample for debugging")
    sp.add_argument("--sample_seed", type=int, default=42)
    sp.set_defaults(func=lambda a: (setattr(a, "max_dwell_sec", None) if a.max_dwell_sec is not None and a.max_dwell_sec < 0 else None, prepare_sessions(a))[1])

    # modelcheck
    sp = sub.add_parser("modelcheck", help="Run finite-trace model checking on prepared sessions")
    sp.add_argument("--sessions_csv", required=True)
    sp.add_argument("--formula_json", required=True)
    sp.add_argument("--out_npz", required=True)
    sp.add_argument("--top_k", type=int, default=None)
    sp.add_argument("--atom_col", default="atom_sequence")
    sp.add_argument("--progress_every", type=int, default=50000)
    sp.set_defaults(func=cmd_modelcheck)

    # train
    sp = sub.add_parser("train", help="Train raw, verification-only, or late-fusion models")
    sp.add_argument("--sessions_csv", required=True)
    sp.add_argument("--splits_dir", required=True, help="Path like /content/electronics_ready/electronics_splits")
    sp.add_argument("--out_dir", required=True)
    sp.add_argument("--dataset_name", default="electronics")
    sp.add_argument("--feature_pack", required=True, choices=[
        "b_raw", "bd_raw", "apc_raw", "adpc_raw",
        "verification_only",
        "b_ltl", "bd_ltl", "apc_ltl", "adpc_ltl",
    ])
    sp.add_argument("--z_npz", default=None, help="Required for verification_only and *_ltl packs")
    sp.add_argument("--length", default="10", help="10, 20, 30, or full")
    sp.add_argument("--full_max_len", type=int, default=512, help="Cap for full-length raw Transformer to avoid extreme padding")
    sp.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    sp.add_argument("--epochs", type=int, default=20)
    sp.add_argument("--patience", type=int, default=4)
    sp.add_argument("--batch_size", type=int, default=256)
    sp.add_argument("--num_workers", type=int, default=2)
    sp.add_argument("--d_model", type=int, default=64)
    sp.add_argument("--z_hidden", type=int, default=64)
    sp.add_argument("--nhead", type=int, default=4)
    sp.add_argument("--num_layers", type=int, default=2)
    sp.add_argument("--dropout", type=float, default=0.1)
    sp.add_argument("--lr", type=float, default=1e-3)
    sp.add_argument("--weight_decay", type=float, default=1e-4)
    sp.add_argument("--grad_clip", type=float, default=1.0)
    sp.add_argument("--monitor", default="auprc", choices=["f1", "mcc", "auc", "auprc", "brier"])
    sp.add_argument("--device", default=None)
    sp.set_defaults(func=cmd_train)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


"""
Quick command templates
-----------------------

# 0) Install dependencies in Colab if needed
# !pip install -q scikit-learn pandas numpy torch

# 1) Prepare session-level dataset with behavior+dwell+product/category columns
python electronics_ltlf_pipeline.py prepare \
  --input /content/electronics/events.csv \
  --out_dir /content/electronics_ready \
  --view_as d \
  --neg_ratio 4 \
  --seeds 42 43 44 45 \
  --max_dwell_sec 1800

# 2) Model-check uploaded behavior-only formulas at K=10
python electronics_ltlf_pipeline.py modelcheck \
  --sessions_csv /content/electronics_ready/electronics_sessions_ltl_ready.csv \
  --formula_json /content/Z_B_semantic_unique_top20_formulas.json \
  --out_npz /content/electronics_ready/Z_B_top10_electronics.npz \
  --top_k 10

# 3) Model-check K=20 robustness
python electronics_ltlf_pipeline.py modelcheck \
  --sessions_csv /content/electronics_ready/electronics_sessions_ltl_ready.csv \
  --formula_json /content/Z_B_semantic_unique_top20_formulas.json \
  --out_npz /content/electronics_ready/Z_B_top20_electronics.npz \
  --top_k 20

# 4) Train behavior-only baselines and fusion
python electronics_ltlf_pipeline.py train \
  --sessions_csv /content/electronics_ready/electronics_sessions_ltl_ready.csv \
  --splits_dir /content/electronics_ready/electronics_splits \
  --out_dir /content/electronics_runs/B10_raw \
  --feature_pack b_raw --length 10

python electronics_ltlf_pipeline.py train \
  --sessions_csv /content/electronics_ready/electronics_sessions_ltl_ready.csv \
  --splits_dir /content/electronics_ready/electronics_splits \
  --out_dir /content/electronics_runs/Bfull_raw \
  --feature_pack b_raw --length full --full_max_len 512

python electronics_ltlf_pipeline.py train \
  --sessions_csv /content/electronics_ready/electronics_sessions_ltl_ready.csv \
  --splits_dir /content/electronics_ready/electronics_splits \
  --out_dir /content/electronics_runs/ZB10_only \
  --feature_pack verification_only \
  --z_npz /content/electronics_ready/Z_B_top10_electronics.npz \
  --length 10

python electronics_ltlf_pipeline.py train \
  --sessions_csv /content/electronics_ready/electronics_sessions_ltl_ready.csv \
  --splits_dir /content/electronics_ready/electronics_splits \
  --out_dir /content/electronics_runs/B10_ZB10 \
  --feature_pack b_ltl \
  --z_npz /content/electronics_ready/Z_B_top10_electronics.npz \
  --length 10
"""
