#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

PACK_TO_COL = {"B": "ltlf_B", "BD": "ltlf_BD", "BP": "ltlf_BP", "BDP": "ltlf_BDP"}

VOCAB_ORDER = {
    "b": 0, "d": 1, "a": 2, "r": 3,
    "dl": 4, "dm": 5, "dh": 6,
    "hp": 7, "np": 8, "sp": 9, "pc": 10,
    "fp": 11, "rp": 12, "p2": 13, "p3": 14,
    "dp1": 15, "dp2": 16, "dp3": 17,
    "hc": 18, "sc": 19, "cc": 20, "rc": 21,
}


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def clone_repo_if_needed(tltlf_repo: Path) -> None:
    if (tltlf_repo / "train.py").exists() and (tltlf_repo / "interpret.py").exists():
        log(f"Using existing TLTLf repo: {tltlf_repo}")
        return
    ensure_dir(tltlf_repo.parent)
    log(f"Cloning TLTLf repo into: {tltlf_repo}")
    subprocess.run(["git", "clone", "https://github.com/a79461378945/TLTLf", str(tltlf_repo)], check=True)


def patch_tltlf_cuda(tltlf_repo: Path) -> None:
    replacement = 'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")'
    for name in ["train.py", "data.py", "interpret.py"]:
        p = tltlf_repo / name
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        old = 'device = torch.device("cpu")'
        if old in txt and replacement not in txt:
            txt = txt.replace(old, replacement)
            p.write_text(txt, encoding="utf-8")
            log(f"Patched CUDA fallback in {p}")
        else:
            log(f"No CUDA patch needed for {p}")


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


def load_vocab_mapping(prep_dir: Path) -> Dict[str, Any]:
    candidates = [
        prep_dir / "cosmetics_dec_vocab_and_symbol_maps.json",
        prep_dir / "vocab_and_symbol_maps.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    log("Warning: vocab/symbol mapping JSON not found. Readable formula mapping will be limited.")
    return {}


def seq_to_trace(seq: Any, trace_tail_len: int | None) -> List[List[str]]:
    toks = str(seq).split()
    if trace_tail_len is not None and trace_tail_len > 0:
        toks = toks[-trace_tail_len:]
    return [[t] for t in toks if t and t != "nan"]


def sample_split(df: pd.DataFrame, label_col: str, sample_pos: int, sample_neg: int, seed: int) -> pd.DataFrame:
    pos = df[df[label_col] == 1].copy()
    neg = df[df[label_col] == 0].copy()
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError(f"Need both positive and negative samples. pos={len(pos)} neg={len(neg)}")
    n_pos = min(sample_pos, len(pos))
    n_neg = min(sample_neg, len(neg))
    pos_s = pos.sample(n=n_pos, random_state=seed)
    neg_s = neg.sample(n=n_neg, random_state=seed + 17)
    out = pd.concat([pos_s, neg_s], axis=0)
    return out.sample(frac=1.0, random_state=seed + 31).reset_index(drop=True)


def collect_vocab_from_df(df: pd.DataFrame, trace_col: str) -> List[str]:
    vocab = set()
    for seq in df[trace_col].dropna().astype(str):
        for t in seq.split():
            if t and t != "nan":
                vocab.add(t)
    return sorted(vocab)


def build_tltlf_json(df: pd.DataFrame, trace_col: str, label_col: str, vocab: List[str], trace_tail_len: int) -> Dict[str, Any]:
    pos_traces, neg_traces = [], []
    for _, row in df.iterrows():
        tr = seq_to_trace(row[trace_col], trace_tail_len)
        if len(tr) == 0:
            continue
        if int(row[label_col]) == 1:
            pos_traces.append(tr)
        else:
            neg_traces.append(tr)
    return {"vocab": vocab, "traces_pos": pos_traces, "traces_neg": neg_traces}


def formula_to_str(f: Any) -> str:
    if isinstance(f, str):
        return f
    if isinstance(f, list):
        f = tuple(f)
    if not isinstance(f, tuple) or len(f) == 0:
        return str(f)
    op = f[0]
    if op in ["!", "not"] and len(f) == 2:
        return f"!({formula_to_str(f[1])})"
    if op == "X" and len(f) == 2:
        return f"X({formula_to_str(f[1])})"
    if op in ["&&", "and", "&"] and len(f) == 3:
        return f"({formula_to_str(f[1])} && {formula_to_str(f[2])})"
    if op == "U" and len(f) == 3:
        return f"({formula_to_str(f[1])} U {formula_to_str(f[2])})"
    return str(f)


def replace_atoms_with_readable(formula: Any, reverse_map: Dict[str, str]) -> str:
    if isinstance(formula, str):
        return reverse_map.get(formula, formula)
    if isinstance(formula, list):
        formula = tuple(formula)
    if not isinstance(formula, tuple) or len(formula) == 0:
        return str(formula)
    op = formula[0]
    if op in ["!", "not"] and len(formula) == 2:
        return f"!({replace_atoms_with_readable(formula[1], reverse_map)})"
    if op == "X" and len(formula) == 2:
        return f"X({replace_atoms_with_readable(formula[1], reverse_map)})"
    if op in ["&&", "and", "&"] and len(formula) == 3:
        return f"({replace_atoms_with_readable(formula[1], reverse_map)} && {replace_atoms_with_readable(formula[2], reverse_map)})"
    if op == "U" and len(formula) == 3:
        return f"({replace_atoms_with_readable(formula[1], reverse_map)} U {replace_atoms_with_readable(formula[2], reverse_map)})"
    return str(formula)


def semantic_vector(ltlf_obj: Any, traces: List[List[List[str]]], prefix: str) -> List[int]:
    out = []
    for i, tr in enumerate(traces):
        try:
            out.append(int(bool(ltlf_obj.pathCheck(tr, f"{prefix}{i}"))))
        except RecursionError:
            out.append(-1)
        except Exception:
            out.append(-1)
    return out


def evaluate_formula(LTLf_cls: Any, formula_tree: Any, vocab: List[str], pos_traces, neg_traces) -> Dict[str, Any]:
    ltlf = LTLf_cls(vocab, formula_tree)
    pos_vec = semantic_vector(ltlf, pos_traces, "pos")
    neg_vec = semantic_vector(ltlf, neg_traces, "neg")
    if any(v < 0 for v in pos_vec + neg_vec):
        return {"valid": False, "semantic_key": None}
    pos_support = float(np.mean(pos_vec)) if pos_vec else 0.0
    neg_support = float(np.mean(neg_vec)) if neg_vec else 0.0
    balanced_accuracy = 0.5 * (pos_support + (1.0 - neg_support))
    support_gap = pos_support - neg_support
    semantic_key = "".join(map(str, pos_vec)) + "|" + "".join(map(str, neg_vec))
    return {
        "valid": True,
        "pos_support": pos_support,
        "neg_support": neg_support,
        "support_gap": support_gap,
        "balanced_accuracy": balanced_accuracy,
        "semantic_key": semantic_key,
    }


def import_tltlf_modules(tltlf_repo: Path):
    sys.path.insert(0, str(tltlf_repo))
    for mod in ["train", "interpret", "data", "model"]:
        if mod in sys.modules:
            del sys.modules[mod]
    train_mod = importlib.import_module("train")
    interpret_mod = importlib.import_module("interpret")
    return train_mod, interpret_mod


def run_one_pack(
    pack: str,
    sessions_csv: Path,
    split_dir: Path,
    prepared_dir: Path,
    out_dir: Path,
    split_seed: int,
    formula_seed: int,
    sample_pos: int,
    sample_neg: int,
    eval_pos: int,
    eval_neg: int,
    trace_tail_len: int,
    top_k: int,
    interpret_max_num: int,
    netlen: int,
    epoch: int,
    lr: float,
    batch_size: int,
    a1: float,
    a2: float,
    a3: float,
    label_col: str,
    train_mod: Any,
    interpret_mod: Any,
    vocab_info: Dict[str, Any],
) -> pd.DataFrame:
    if pack not in PACK_TO_COL:
        raise ValueError(f"Unknown pack={pack}. Choose from {list(PACK_TO_COL)}")
    trace_col = PACK_TO_COL[pack]

    log("\n" + "=" * 80)
    log(f"Pack {pack}: trace_col={trace_col}")
    log("=" * 80)

    pack_dir = ensure_dir(out_dir / pack)
    data_dir = ensure_dir(pack_dir / "data")
    model_dir = ensure_dir(pack_dir / "model")
    log_dir = ensure_dir(pack_dir / "log")

    train_path = split_dir / f"seed_{split_seed}" / "train.csv"
    val_path = split_dir / f"seed_{split_seed}" / "val.csv"
    train_df = pd.read_csv(train_path, dtype={"session_id_hash": str})
    val_df = pd.read_csv(val_path, dtype={"session_id_hash": str})

    train_sample = sample_split(train_df, label_col, sample_pos, sample_neg, formula_seed)
    eval_sample = sample_split(val_df, label_col, eval_pos, eval_neg, formula_seed + 101)

    sessions = pd.read_csv(sessions_csv, usecols=[trace_col])
    vocab = collect_vocab_from_df(sessions, trace_col)
    if len(vocab) == 0:
        raise ValueError(f"Empty vocab for {pack}")

    train_json = build_tltlf_json(train_sample, trace_col, label_col, vocab, trace_tail_len)
    test_json = build_tltlf_json(eval_sample, trace_col, label_col, vocab, trace_tail_len)

    train_json_path = data_dir / "train.json"
    test_json_path = data_dir / "test.json"
    with open(train_json_path, "w", encoding="utf-8") as f:
        json.dump(train_json, f)
    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(test_json, f)
    train_sample.to_csv(data_dir / "train_sample.csv", index=False)
    eval_sample.to_csv(data_dir / "eval_sample.csv", index=False)

    log(f"vocab size: {len(vocab)}")
    log(f"train pos/neg traces: {len(train_json['traces_pos'])}/{len(train_json['traces_neg'])}")
    log(f"eval  pos/neg traces: {len(test_json['traces_pos'])}/{len(test_json['traces_neg'])}")
    log(f"trace_tail_len: {trace_tail_len}")

    model_path = model_dir / "tmodel"
    tb_log_path = log_dir / "tlog"
    set_all_seeds(formula_seed)

    train_args = train_mod.get_train_argument([
        "--train_file", str(train_json_path),
        "--save_model", str(model_path),
        "--log_file", str(tb_log_path),
        "--netlen", str(netlen),
        "--epoch", str(epoch),
        "--lr", str(lr),
        "--batch_size", str(batch_size),
        "--a1", str(a1),
        "--a2", str(a2),
        "--a3", str(a3),
    ])

    t0 = time.time()
    train_time = train_mod.train(train_args)
    wall_train_time = time.time() - t0
    log(f"Training finished. train_time={train_time:.2f}s wall={wall_train_time:.2f}s")

    t0 = time.time()
    candidates = interpret_mod.matrix2ltl(str(model_path), str(test_json_path), interpret_max_num)
    interpret_time = time.time() - t0
    log(f"Interpreted candidates: {len(candidates)} in {interpret_time:.2f}s")

    reverse_map: Dict[str, str] = {}
    symbol_maps = vocab_info.get("symbol_maps", {})
    if pack in symbol_maps:
        reverse_map = {v: k for k, v in symbol_maps[pack].items()}

    rows: List[Dict[str, Any]] = []
    seen_semantics = set()

    for rank, formula_tree in enumerate(candidates, start=1):
        metrics = evaluate_formula(interpret_mod.LTLf, formula_tree, vocab, test_json["traces_pos"], test_json["traces_neg"])
        if not metrics.get("valid"):
            continue
        sem_key = metrics["semantic_key"]
        semantic_duplicate = sem_key in seen_semantics
        if not semantic_duplicate:
            seen_semantics.add(sem_key)
        rows.append({
            "pack": pack,
            "rank_from_tltlf": rank,
            "formula_tree": json.dumps(formula_tree),
            "formula_str": formula_to_str(formula_tree),
            "formula_readable": replace_atoms_with_readable(formula_tree, reverse_map),
            "pos_support": metrics["pos_support"],
            "neg_support": metrics["neg_support"],
            "support_gap": metrics["support_gap"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "semantic_duplicate": semantic_duplicate,
            "trace_tail_len": trace_tail_len,
            "top_k_requested": top_k,
            "interpret_max_num": interpret_max_num,
            "netlen": netlen,
            "epoch": epoch,
            "sample_pos": sample_pos,
            "sample_neg": sample_neg,
            "eval_pos": eval_pos,
            "eval_neg": eval_neg,
            "formula_seed": formula_seed,
            "split_seed": split_seed,
            "train_time_sec": float(train_time),
            "wall_train_time_sec": float(wall_train_time),
            "interpret_time_sec": float(interpret_time),
            "vocab_size": len(vocab),
        })

    candidates_df = pd.DataFrame(rows)
    if len(candidates_df) == 0:
        raise RuntimeError(f"No valid interpreted formulas for pack={pack}")

    candidates_df = candidates_df.sort_values(
        ["balanced_accuracy", "support_gap", "pos_support"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    candidates_df.to_csv(pack_dir / f"{pack}_all_candidate_formulas.csv", index=False)

    top_df = candidates_df[candidates_df["semantic_duplicate"] == False].head(top_k).copy().reset_index(drop=True)
    top_df.insert(0, "formula_id", [f"T{i:02d}" for i in range(1, len(top_df) + 1)])
    top_csv = pack_dir / f"{pack}_semantic_unique_top{top_k}_formulas.csv"
    top_json = pack_dir / f"{pack}_semantic_unique_top{top_k}_formulas.json"
    top_df.to_csv(top_csv, index=False)
    with open(top_json, "w", encoding="utf-8") as f:
        json.dump(top_df.to_dict(orient="records"), f, indent=2)

    log("\nTop semantic-unique formulas:")
    cols = ["formula_id", "rank_from_tltlf", "formula_str", "pos_support", "neg_support", "support_gap", "balanced_accuracy"]
    log(str(top_df[cols].to_string(index=False)))
    log(f"Saved top CSV : {top_csv}")
    log(f"Saved top JSON: {top_json}")
    return top_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Learn top-K TLTLf formulas for cosmetics packs.")
    p.add_argument("--base", type=str, default="/content/drive/MyDrive/ltlf_cosmetics_experiments")
    p.add_argument("--prepared_dir", type=str, default=None)
    p.add_argument("--sessions_csv", type=str, required=True)
    p.add_argument("--split_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--tltlf_repo", type=str, default="/content/TLTLf")
    p.add_argument("--packs", nargs="+", default=["B", "BD", "BP", "BDP"])
    p.add_argument("--sample_pos", type=int, default=4000)
    p.add_argument("--sample_neg", type=int, default=4000)
    p.add_argument("--eval_pos", type=int, default=4000)
    p.add_argument("--eval_neg", type=int, default=4000)
    p.add_argument("--trace_tail_len", type=int, default=20)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--interpret_max_num", type=int, default=300)
    p.add_argument("--formula_seed", type=int, default=42)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--netlen", type=int, default=10)
    p.add_argument("--epoch", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--a1", type=float, default=0.1)
    p.add_argument("--a2", type=float, default=0.1)
    p.add_argument("--a3", type=float, default=0.1)
    p.add_argument("--label_col", type=str, default="purchase")
    p.add_argument("--clone_repo", action="store_true")
    p.add_argument("--patch_cuda", action="store_true")
    p.add_argument("--stop_on_error", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    prepared_dir = Path(args.prepared_dir) if args.prepared_dir else Path(args.sessions_csv).parent
    sessions_csv = Path(args.sessions_csv)
    split_dir = Path(args.split_dir)
    out_dir = ensure_dir(Path(args.out_dir))
    tltlf_repo = Path(args.tltlf_repo)

    log("=" * 80)
    log("Learn TLTLf formulas for Cosmetics dataset")
    log("=" * 80)
    log(f"prepared_dir      : {prepared_dir}")
    log(f"sessions_csv      : {sessions_csv}")
    log(f"split_dir         : {split_dir}")
    log(f"out_dir           : {out_dir}")
    log(f"tltlf_repo        : {tltlf_repo}")
    log(f"packs             : {args.packs}")
    log(f"trace_tail_len    : {args.trace_tail_len}")
    log(f"top_k             : {args.top_k}")
    log(f"sample_pos/neg    : {args.sample_pos}/{args.sample_neg}")
    log(f"eval_pos/neg      : {args.eval_pos}/{args.eval_neg}")
    log(f"formula_seed      : {args.formula_seed}")
    log(f"split_seed        : {args.split_seed}\n")

    if args.clone_repo:
        clone_repo_if_needed(tltlf_repo)
    if args.patch_cuda:
        patch_tltlf_cuda(tltlf_repo)
    if not (tltlf_repo / "train.py").exists():
        raise FileNotFoundError(f"TLTLf repo not found or incomplete: {tltlf_repo}")

    vocab_info = load_vocab_mapping(prepared_dir)
    train_mod, interpret_mod = import_tltlf_modules(tltlf_repo)

    all_top = []
    errors = []
    for pack in args.packs:
        try:
            top_df = run_one_pack(
                pack=pack,
                sessions_csv=sessions_csv,
                split_dir=split_dir,
                prepared_dir=prepared_dir,
                out_dir=out_dir,
                split_seed=args.split_seed,
                formula_seed=args.formula_seed,
                sample_pos=args.sample_pos,
                sample_neg=args.sample_neg,
                eval_pos=args.eval_pos,
                eval_neg=args.eval_neg,
                trace_tail_len=args.trace_tail_len,
                top_k=args.top_k,
                interpret_max_num=args.interpret_max_num,
                netlen=args.netlen,
                epoch=args.epoch,
                lr=args.lr,
                batch_size=args.batch_size,
                a1=args.a1,
                a2=args.a2,
                a3=args.a3,
                label_col=args.label_col,
                train_mod=train_mod,
                interpret_mod=interpret_mod,
                vocab_info=vocab_info,
            )
            all_top.append(top_df)
        except Exception as exc:
            msg = f"Pack {pack} failed: {type(exc).__name__}: {exc}"
            log(msg)
            errors.append(msg)
            if args.stop_on_error:
                raise

    if all_top:
        summary = pd.concat(all_top, axis=0, ignore_index=True)
        summary_path = out_dir / f"all_packs_semantic_unique_top{args.top_k}_summary.csv"
        summary.to_csv(summary_path, index=False)
        log("\n" + "=" * 80)
        log("All-pack summary")
        log("=" * 80)
        cols = ["pack", "formula_id", "rank_from_tltlf", "formula_str", "pos_support", "neg_support", "support_gap", "balanced_accuracy"]
        log(str(summary[cols].to_string(index=False)))
        log(f"Saved summary: {summary_path}")

    if errors:
        err_path = out_dir / "errors.txt"
        err_path.write_text("\n".join(errors), encoding="utf-8")
        log("\nSome packs failed:")
        for e in errors:
            log("  " + e)
        log(f"Saved errors: {err_path}")

    log("\nDone.")


if __name__ == "__main__":
    main()
