#!/usr/bin/env python3
"""
learn_cosmetics_tltlf_formulas_gpu.py

Cosmetics version adapted from the previous electronics TLTLf formula-learning script.

Key differences from the failed v5 script:
  - Uses the previous multi-atom trace style, not composite-symbol ltlf_B/ltlf_BD/... columns.
  - Builds TLTLf JSON from prop_BDP / atom_sequence and filters atoms by pack vocab.
  - Supports formula learning on the last L input events via --trace_tail_len.
  - If --patch_cuda is enabled, patches train.py, data.py, model.py, and interpret.py.
    Patching model.py is required; otherwise PyTorch can hit CPU/GPU tensor mismatch.

Expected prepared cosmetics CSV columns:
  - purchase
  - prop_BDP   (preferred; event-level atoms joined by '+', steps joined by spaces)
    or atom_sequence
  - session_id_hash

Packs:
  B   = behavior only
  BD  = behavior + dwell
  BP  = behavior + product/category propositions
  BDP = behavior + dwell + product/category propositions
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


PACKS: Dict[str, List[str]] = {
    # Cosmetics preparation maps view -> d, cart -> a, remove_from_cart -> r.
    # Keep b optional for compatibility with older prepared files.
    "B":   ["b", "d", "a", "r"],
    "BD":  ["b", "d", "a", "r", "dl", "dm", "dh"],
    "BP":  [
        "b", "d", "a", "r",
        "hp", "np", "sp", "pc", "fp", "rp", "p2", "p3", "dp1", "dp2", "dp3",
        "hc", "sc", "cc", "rc",
    ],
    "BDP": [
        "b", "d", "a", "r", "dl", "dm", "dh",
        "hp", "np", "sp", "pc", "fp", "rp", "p2", "p3", "dp1", "dp2", "dp3",
        "hc", "sc", "cc", "rc",
    ],
}


# -----------------------------
# TLTLf repo patching / import
# -----------------------------

def patch_tltlf_cuda(repo_dir: Path) -> None:
    """
    Patch all TLTLf files that define a device.

    Important:
    Earlier we patched train.py/data.py/interpret.py but not model.py.
    model.py defines its own `device`, so leaving it on CPU causes:
      Expected all tensors to be on the same device, but mat2 is on cpu and others on cuda.
    """
    replacement = 'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")'
    for name in ["train.py", "data.py", "model.py", "interpret.py"]:
        p = repo_dir / name
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        changed = False
        if 'device = torch.device("cpu")' in txt:
            txt = txt.replace('device = torch.device("cpu")', replacement)
            changed = True
        # If a previous broken/no-cuda patch exists, normalize it.
        if 'device=torch.device("cpu")' in txt:
            txt = txt.replace('device=torch.device("cpu")', f"device={replacement.split(' = ',1)[1]}")
            changed = True
        if changed:
            p.write_text(txt, encoding="utf-8")
            print(f"Patched CUDA fallback in {p}", flush=True)
        else:
            print(f"No CUDA patch needed for {p}", flush=True)



def patch_interpret_numpy_to_cpu(repo_dir: Path) -> None:
    """
    Patch interpret.py so CUDA tensors are copied to CPU before numpy conversion.

    The TLTLf interpret.py uses patterns like:
      (torch.softmax(parameter['chose_right.weight'], dim=1)+1e-6).numpy()
    so generic .data.numpy() replacement is not enough.
    """
    p = repo_dir / "interpret.py"
    if not p.exists():
        return

    txt = p.read_text(encoding="utf-8", errors="ignore")
    original = txt

    # Exact fixes for the public TLTLf interpret.py
    txt = txt.replace(
        "(torch.softmax(parameter['chose_right.weight'], dim=1)+1e-6).numpy()",
        "(torch.softmax(parameter['chose_right.weight'], dim=1)+1e-6).detach().cpu().numpy()",
    )
    txt = txt.replace(
        "(torch.softmax(parameter['chose_op.weight'], dim=1)+1e-6).numpy()",
        "(torch.softmax(parameter['chose_op.weight'], dim=1)+1e-6).detach().cpu().numpy()",
    )

    # Common fallback patterns, kept safe.
    txt = txt.replace(".data.numpy()", ".detach().cpu().numpy()")
    txt = txt.replace(".weight.numpy()", ".weight.detach().cpu().numpy()")
    txt = txt.replace(".bias.numpy()", ".bias.detach().cpu().numpy()")
    txt = txt.replace(".detach().cpu().detach().cpu().numpy()", ".detach().cpu().numpy()")

    if txt != original:
        p.write_text(txt, encoding="utf-8")
        print(f"Patched CUDA tensor numpy conversion in {p}", flush=True)
    else:
        print(f"No numpy conversion patch applied for {p}; checking current numpy lines:", flush=True)
        for line in txt.splitlines():
            if ".numpy()" in line:
                print("  " + line.strip(), flush=True)


def import_tltlf(repo_dir: Path):
    # Make sure stale CPU modules are not reused in the same process.
    for m in ["train", "data", "model", "interpret"]:
        if m in sys.modules:
            del sys.modules[m]
    sys.path.insert(0, str(repo_dir))
    from train import get_train_argument, train as tltlf_train
    from interpret import matrix2ltl, LTLf
    return get_train_argument, tltlf_train, matrix2ltl, LTLf


# -----------------------------
# Parsing prepared traces
# -----------------------------

def safe_load_list(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    if x is None:
        return []
    try:
        if pd.isna(x):
            return []
    except Exception:
        pass
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            return json.loads(s)
        except Exception:
            try:
                return ast.literal_eval(s)
            except Exception:
                return []
    try:
        return list(x)
    except Exception:
        return []


def prop_sequence_to_trace(x: Any) -> List[List[str]]:
    """
    Parse prop_BDP style:
      "d+dl+hp+fp+dp1+hc d+dm+hp+sp+rp+p2+dp1+hc+sc+rc a+dh+..."
    into:
      [["d","dl","hp","fp","dp1","hc"], ...]
    """
    if x is None:
        return [[]]
    try:
        if pd.isna(x):
            return [[]]
    except Exception:
        pass

    s = str(x).strip()
    if not s:
        return [[]]

    trace: List[List[str]] = []
    for step in s.split():
        atoms = [a for a in re.split(r"[+,]+", step.strip()) if a]
        trace.append(sorted(set(atoms)))
    return trace if trace else [[]]


def atom_sequence_to_trace(x: Any) -> List[List[str]]:
    """
    Parse atom_sequence into list-of-list-of-atoms.
    Supports JSON/list formats from older electronics preparation.
    """
    steps = safe_load_list(x)
    if not steps:
        # Fallback: maybe it is actually prop_BDP style.
        return prop_sequence_to_trace(x)

    trace: List[List[str]] = []
    for step in steps:
        if isinstance(step, list):
            atoms = [str(a) for a in step]
        elif isinstance(step, set):
            atoms = [str(a) for a in step]
        elif isinstance(step, str):
            atoms = [s for s in re.split(r"[+,\s]+", step.strip()) if s]
        else:
            try:
                atoms = [str(a) for a in list(step)]
            except Exception:
                atoms = []
        trace.append(sorted(set(atoms)))
    return trace if trace else [[]]


def filter_trace(
    trace: Sequence[Sequence[str]],
    vocab: Sequence[str],
    trace_tail_len: Optional[int] = None,
) -> List[List[str]]:
    allowed = set(vocab)
    out: List[List[str]] = []
    for step in trace:
        atoms = sorted([a for a in set(map(str, step)) if a in allowed])
        out.append(atoms)

    if not out:
        out = [[]]

    if trace_tail_len is not None and trace_tail_len > 0:
        out = out[-trace_tail_len:]

    return out if out else [[]]


def tree_to_formula(t: Any) -> str:
    """Convert TLTLf tuple/list tree into formula string accepted by later model-checking scripts."""
    if isinstance(t, str):
        return t
    if not isinstance(t, (tuple, list)) or len(t) == 0:
        return str(t)

    op = t[0]
    if op in ["!", "not"]:
        return f"!({tree_to_formula(t[1])})"
    if op in ["X", "N"]:
        return f"X({tree_to_formula(t[1])})"
    if op in ["&&", "&", "and"]:
        return f"({tree_to_formula(t[1])} && {tree_to_formula(t[2])})"
    if op in ["U"]:
        return f"({tree_to_formula(t[1])} U {tree_to_formula(t[2])})"
    if op in ["||", "|", "or"]:
        return f"({tree_to_formula(t[1])} || {tree_to_formula(t[2])})"
    return str(t)


def ensure_atom_column(split_df: pd.DataFrame, sessions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure split_df has an atom_sequence-like column.

    Priority:
      1. Existing atom_sequence
      2. Existing prop_BDP
      3. Merge prop_BDP from sessions_csv by session_id_hash
      4. Merge atom_sequence from sessions_csv by session_id_hash
    """
    if "atom_sequence" in split_df.columns:
        return split_df
    if "prop_BDP" in split_df.columns:
        split_df = split_df.copy()
        split_df["atom_sequence"] = split_df["prop_BDP"]
        return split_df

    if "session_id_hash" not in split_df.columns:
        raise ValueError("split CSV has no atom_sequence/prop_BDP and no session_id_hash.")

    if "prop_BDP" in sessions_df.columns:
        cols = ["session_id_hash", "prop_BDP"]
        merged = split_df.merge(sessions_df[cols], on="session_id_hash", how="left")
        if merged["prop_BDP"].isna().any():
            missing = int(merged["prop_BDP"].isna().sum())
            raise ValueError(f"Failed to merge prop_BDP for {missing} rows.")
        merged["atom_sequence"] = merged["prop_BDP"]
        return merged

    if "atom_sequence" in sessions_df.columns:
        cols = ["session_id_hash", "atom_sequence"]
        merged = split_df.merge(sessions_df[cols], on="session_id_hash", how="left")
        if merged["atom_sequence"].isna().any():
            missing = int(merged["atom_sequence"].isna().sum())
            raise ValueError(f"Failed to merge atom_sequence for {missing} rows.")
        return merged

    raise ValueError("Neither split nor sessions CSV contains atom_sequence or prop_BDP.")


def sample_by_label(
    df: pd.DataFrame,
    n_per_class: int,
    label_col: str,
    seed: int,
    allow_smaller: bool = False,
) -> pd.DataFrame:
    pos = df[df[label_col].astype(int) == 1]
    neg = df[df[label_col].astype(int) == 0]

    if not allow_smaller:
        if len(pos) < n_per_class or len(neg) < n_per_class:
            raise ValueError(
                f"Not enough samples: pos={len(pos)}, neg={len(neg)}, requested={n_per_class} each. "
                f"Use --allow_smaller if this is intentional."
            )

    n_pos = min(len(pos), n_per_class)
    n_neg = min(len(neg), n_per_class)

    out = pd.concat(
        [
            pos.sample(n=n_pos, random_state=seed),
            neg.sample(n=n_neg, random_state=seed + 13),
        ],
        axis=0,
    ).sample(frac=1.0, random_state=seed + 29).reset_index(drop=True)

    return out


def build_tltlf_json(
    df: pd.DataFrame,
    vocab: Sequence[str],
    label_col: str = "purchase",
    atom_col: str = "atom_sequence",
    trace_tail_len: Optional[int] = None,
) -> Dict[str, Any]:
    traces_pos, traces_neg = [], []

    for _, row in df.iterrows():
        raw_trace = atom_sequence_to_trace(row[atom_col])
        trace = filter_trace(raw_trace, vocab, trace_tail_len=trace_tail_len)

        if int(row[label_col]) == 1:
            traces_pos.append(trace)
        else:
            traces_neg.append(trace)

    return {
        "vocab": list(vocab),
        "traces_pos": traces_pos,
        "traces_neg": traces_neg,
        # Present only to satisfy repo loaders that expect a target tree;
        # it is not used when --test_correctness=0.
        "ltlftree": "true",
    }


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def evaluate_formula_tree(
    LTLf_cls,
    vocab: Sequence[str],
    formula_tree: Any,
    val_obj: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    try:
        ltlf = LTLf_cls(list(vocab), formula_tree)
        pos_marks, neg_marks = ltlf.evaluate(val_obj["traces_pos"], val_obj["traces_neg"])
        pos_arr = np.asarray(pos_marks, dtype=np.uint8)
        neg_arr = np.asarray(neg_marks, dtype=np.uint8)
        vector = tuple(pos_arr.tolist() + neg_arr.tolist())

        pos_support = float(pos_arr.mean()) if len(pos_arr) else 0.0
        neg_support = float(neg_arr.mean()) if len(neg_arr) else 0.0
        bal_acc = 0.5 * (pos_support + (1.0 - neg_support))

        return {
            "pos_support": pos_support,
            "neg_support": neg_support,
            "support_gap": pos_support - neg_support,
            "balanced_accuracy": bal_acc,
            "satisfaction_vector": vector,
        }
    except Exception:
        return None


@dataclass
class PackResult:
    pack: str
    selected_json: Path
    selected_csv: Path
    all_candidates_csv: Path
    train_json: Path
    val_json: Path


def learn_one_pack(
    *,
    pack: str,
    vocab: Sequence[str],
    repo_dir: Path,
    out_dir: Path,
    train_obj: Dict[str, Any],
    val_obj: Dict[str, Any],
    netlen: int,
    epochs: int,
    lr: float,
    batch_size: int,
    restarts: int,
    max_candidates: int,
    select_k: int,
    seed: int,
    reuse_models: bool = False,
) -> PackResult:
    pack_dir = out_dir / pack
    data_dir = pack_dir / "data"
    model_dir = pack_dir / "model"
    log_dir = pack_dir / "log"

    for d in [data_dir, model_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    train_json = data_dir / "train.json"
    val_json = data_dir / "val.json"

    with open(train_json, "w") as f:
        json.dump(train_obj, f)

    with open(val_json, "w") as f:
        json.dump(val_obj, f)

    get_train_argument, tltlf_train, matrix2ltl, LTLf_cls = import_tltlf(repo_dir)

    all_rows: List[Dict[str, Any]] = []
    seen_formula_str: Set[str] = set()

    for r in range(restarts):
        run_seed = seed + 1009 * r
        set_all_seeds(run_seed)

        model_path = model_dir / f"tmodel_restart{r:02d}.pt"
        this_log_dir = log_dir / f"restart{r:02d}"

        args = get_train_argument(
            [
                "--train_file", str(train_json),
                "--save_model", str(model_path),
                "--log_file", str(this_log_dir),
                "--netlen", str(netlen),
                "--epoch", str(epochs),
                "--lr", str(lr),
                "--batch_size", str(batch_size),
                "--test_correctness", "0",
            ]
        )

        if reuse_models and model_path.exists():
            print(f"\n=== Reusing existing model pack={pack} restart={r}: {model_path} ===", flush=True)
        else:
            print(f"\n=== Training pack={pack} restart={r} seed={run_seed} ===", flush=True)
            t0 = time.time()
            elapsed = tltlf_train(args)
            print(
                f"Training finished: pack={pack}, restart={r}, "
                f"time={elapsed:.2f}s, wall={time.time() - t0:.2f}s",
                flush=True,
            )

        print(f"Interpreting model: {model_path}", flush=True)
        try:
            formula_trees = matrix2ltl(str(model_path), str(val_json), max_candidates)
        except Exception as e:
            print(f"[WARN] matrix2ltl failed for pack={pack} restart={r}: {e}", flush=True)
            continue

        for local_rank, ftree in enumerate(formula_trees, start=1):
            fstr = tree_to_formula(ftree)
            if fstr in seen_formula_str:
                continue

            eval_info = evaluate_formula_tree(LTLf_cls, vocab, ftree, val_obj)
            if eval_info is None:
                continue

            seen_formula_str.add(fstr)

            row = {
                "pack": pack,
                "restart": r,
                "restart_seed": run_seed,
                "rank_from_tltlf": local_rank,
                "formula_tree": repr(ftree),
                "formula_str": fstr,
                "pos_support": eval_info["pos_support"],
                "neg_support": eval_info["neg_support"],
                "support_gap": eval_info["support_gap"],
                "balanced_accuracy": eval_info["balanced_accuracy"],
                "_satisfaction_vector": eval_info["satisfaction_vector"],
            }
            all_rows.append(row)

    if not all_rows:
        raise RuntimeError(f"No candidate formulas extracted for pack={pack}.")

    # Sort first, then semantic-unique by validation satisfaction vector.
    all_rows.sort(
        key=lambda x: (
            x["balanced_accuracy"],
            abs(x["support_gap"]),
            x["pos_support"],
            -x["rank_from_tltlf"],
        ),
        reverse=True,
    )

    selected: List[Dict[str, Any]] = []
    seen_vectors: Set[Tuple[int, ...]] = set()

    for row in all_rows:
        vec = tuple(row["_satisfaction_vector"])
        if vec in seen_vectors:
            continue
        seen_vectors.add(vec)

        clean = {k: v for k, v in row.items() if k != "_satisfaction_vector"}
        clean["formula_id"] = f"T{len(selected) + 1:02d}"
        selected.append(clean)

        if len(selected) >= select_k:
            break

    all_candidates_csv = pack_dir / f"{pack}_all_candidates.csv"
    pd.DataFrame(
        [{k: v for k, v in r.items() if k != "_satisfaction_vector"} for r in all_rows]
    ).to_csv(all_candidates_csv, index=False)

    selected_csv = pack_dir / f"{pack}_semantic_unique_top{select_k}_formulas.csv"
    pd.DataFrame(selected).to_csv(selected_csv, index=False)

    selected_json = pack_dir / f"Z_{pack}_semantic_unique_top{select_k}_formulas.json"
    out_obj = {
        "dataset": "cosmetics_dec",
        "pack": pack,
        "selection": "semantic_unique_by_validation_satisfaction_vector",
        "repo": "https://github.com/a79461378945/TLTLf",
        "vocab": list(vocab),
        "netlen": netlen,
        "sample_per_class": {
            "train_pos": len(train_obj["traces_pos"]),
            "train_neg": len(train_obj["traces_neg"]),
            "val_pos": len(val_obj["traces_pos"]),
            "val_neg": len(val_obj["traces_neg"]),
        },
        "max_candidates_per_restart": max_candidates,
        "restarts": restarts,
        "selected_K": len(selected),
        "formulas": selected,
    }

    with open(selected_json, "w") as f:
        json.dump(out_obj, f, indent=2)

    print(f"\n=== Selected semantic-unique formulas for {pack} ===", flush=True)
    for row in selected:
        print(
            f"{row['formula_id']} bal={row['balanced_accuracy']:.4f} "
            f"pos={row['pos_support']:.4f} neg={row['neg_support']:.4f} :: {row['formula_str']}",
            flush=True,
        )

    return PackResult(pack, selected_json, selected_csv, all_candidates_csv, train_json, val_json)


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("--repo_dir", type=Path, required=True)
    p.add_argument("--sessions_csv", type=Path, required=True)
    p.add_argument("--splits_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)

    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--sample_per_class", type=int, default=4000)
    p.add_argument("--val_per_class", type=int, default=4000)
    p.add_argument("--packs", nargs="+", default=["B", "BD", "BP", "BDP"], choices=list(PACKS.keys()))

    p.add_argument("--trace_tail_len", type=int, default=20)
    p.add_argument("--netlen", type=int, default=10)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--restarts", type=int, default=1)
    p.add_argument("--max_candidates", type=int, default=300)
    p.add_argument("--select_k", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--label_col", default="purchase")
    p.add_argument("--atom_col", default="atom_sequence")
    p.add_argument("--allow_smaller", action="store_true")
    p.add_argument("--patch_cuda", action="store_true")
    p.add_argument("--reuse_models", action="store_true", help="Skip training when model restart file already exists; only run interpretation/selection.")

    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("Learn TLTLf formulas for Cosmetics dataset -- electronics-style multi-atom GPU script v3", flush=True)
    print("=" * 80, flush=True)
    print(f"repo_dir        : {args.repo_dir}", flush=True)
    print(f"sessions_csv    : {args.sessions_csv}", flush=True)
    print(f"splits_dir      : {args.splits_dir}", flush=True)
    print(f"out_dir         : {args.out_dir}", flush=True)
    print(f"packs           : {args.packs}", flush=True)
    print(f"trace_tail_len  : {args.trace_tail_len}", flush=True)
    print(f"sample/val class: {args.sample_per_class}/{args.val_per_class}", flush=True)
    print(f"select_k        : {args.select_k}", flush=True)
    print(f"restarts        : {args.restarts}", flush=True)
    print(f"patch_cuda      : {args.patch_cuda}", flush=True)
    print(f"reuse_models    : {args.reuse_models}", flush=True)

    if args.patch_cuda:
        patch_tltlf_cuda(args.repo_dir)
        patch_interpret_numpy_to_cpu(args.repo_dir)

    sessions_df = pd.read_csv(args.sessions_csv)

    split_dir = args.splits_dir / f"seed_{args.split_seed}"
    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"

    if not train_csv.exists():
        raise FileNotFoundError(train_csv)
    if not val_csv.exists():
        raise FileNotFoundError(val_csv)

    train_df = ensure_atom_column(pd.read_csv(train_csv), sessions_df)
    val_df = ensure_atom_column(pd.read_csv(val_csv), sessions_df)

    train_sample = sample_by_label(
        train_df,
        args.sample_per_class,
        args.label_col,
        args.seed,
        allow_smaller=args.allow_smaller,
    )

    val_sample = sample_by_label(
        val_df,
        args.val_per_class,
        args.label_col,
        args.seed + 99,
        allow_smaller=True,
    )

    print("Sampling summary:", flush=True)
    print(" train:", train_sample[args.label_col].astype(int).value_counts().to_dict(), flush=True)
    print(" val:  ", val_sample[args.label_col].astype(int).value_counts().to_dict(), flush=True)

    manifest = {
        "dataset": "cosmetics_dec",
        "sessions_csv": str(args.sessions_csv),
        "splits_dir": str(args.splits_dir),
        "split_seed": args.split_seed,
        "trace_tail_len": args.trace_tail_len,
        "sample_per_class": args.sample_per_class,
        "val_per_class": args.val_per_class,
        "packs": args.packs,
        "netlen": args.netlen,
        "epochs": args.epochs,
        "restarts": args.restarts,
        "select_k": args.select_k,
        "repo_dir": str(args.repo_dir),
        "outputs": {},
    }

    results = []

    for pack in args.packs:
        vocab = PACKS[pack]
        print(f"\n\n########## PACK {pack}: vocab={vocab} ##########", flush=True)

        train_obj = build_tltlf_json(
            train_sample,
            vocab,
            args.label_col,
            args.atom_col,
            trace_tail_len=args.trace_tail_len,
        )

        val_obj = build_tltlf_json(
            val_sample,
            vocab,
            args.label_col,
            args.atom_col,
            trace_tail_len=args.trace_tail_len,
        )

        print(
            f"TLTLf JSON traces: train pos/neg={len(train_obj['traces_pos'])}/{len(train_obj['traces_neg'])}, "
            f"val pos/neg={len(val_obj['traces_pos'])}/{len(val_obj['traces_neg'])}",
            flush=True,
        )

        res = learn_one_pack(
            pack=pack,
            vocab=vocab,
            repo_dir=args.repo_dir,
            out_dir=args.out_dir,
            train_obj=train_obj,
            val_obj=val_obj,
            netlen=args.netlen,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            restarts=args.restarts,
            max_candidates=args.max_candidates,
            select_k=args.select_k,
            seed=args.seed,
            reuse_models=args.reuse_models,
        )

        results.append(res)
        manifest["outputs"][pack] = {
            "selected_json": str(res.selected_json),
            "selected_csv": str(res.selected_csv),
            "all_candidates_csv": str(res.all_candidates_csv),
            "train_json": str(res.train_json),
            "val_json": str(res.val_json),
        }

    with open(args.out_dir / "formula_learning_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Combined summary CSV
    summary_frames = []
    for res in results:
        summary_frames.append(pd.read_csv(res.selected_csv))
    if summary_frames:
        summary = pd.concat(summary_frames, ignore_index=True)
        summary.to_csv(args.out_dir / f"all_packs_semantic_unique_top{args.select_k}_summary.csv", index=False)

    print("\n\nDONE. Selected formula JSON files:", flush=True)
    for res in results:
        print(f"  {res.pack}: {res.selected_json}", flush=True)


if __name__ == "__main__":
    main()
