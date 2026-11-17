#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integrated PyTorch experiment for LTL/LTLf model checking + neural training.

This script does BOTH steps in one run:
  1. formula JSON + clickstream traces -> binary model checking OR automaton-state/progression representation -> Z matrix
  2. raw sequence / Z / raw+Z -> PyTorch training -> metrics + paper tables

Default experiment settings:
  - MC uses FULL trace by default: --mc_window 0
  - For prefix-aware experiments, use --mc_window K --mc_truncation first --raw_truncation first
  - For automaton-state features, add --ltl_representation automaton
  - Raw baselines: Raw-10, Raw-20, Raw-30, Raw-Full
  - Symbolic baseline: Verification-only
  - Neural-symbolic: Raw-10+LTL, Raw-20+LTL, Raw-30+LTL
  - No manual LTL rules are used.

Useful MC settings:
  FULL trace:       --mc_window 0
  last 30 events:   --mc_window 30 --mc_truncation last
  first 30 prefix:  --mc_window 30 --mc_truncation first

Formula modes:
  fixed:
    all seeds use the same formula JSON, e.g.
    --formula_mode fixed --formula_json /content/top20_semantic_unique_formulas.json

  seed_specific:
    each seed uses its own formula JSON from formula_json_pattern, e.g.
    /content/aligned_splits_v1/seed_{seed}/window_last30/formulas/top20_semantic_unique_formulas.json

Outputs:
  out_dir/run_name/
    all_runs.csv
    summary_uncertainty.csv
    summary_cost_adjusted.csv
    pareto_front.csv
    model_checking_by_seed.csv
    sequence_length_stats.csv
    paper_table_main.tex / .csv
    paper_table_cost.tex / .csv
    paper_table_wide.tex / .csv
    seed_42/Z_seed42_*.npz / .csv / metadata
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    cohen_kappa_score,
    confusion_matrix,
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ============================================================
# Formula loading and finite LTLf evaluation
# ============================================================

def jsonable_to_formula(x):
    if isinstance(x, list):
        return tuple(jsonable_to_formula(v) for v in x)
    return x


def formula_to_jsonable(f):
    if isinstance(f, tuple):
        return [formula_to_jsonable(x) for x in f]
    return f


def formula_to_str(f) -> str:
    if isinstance(f, str):
        return f
    if isinstance(f, (tuple, list)):
        if len(f) == 2:
            return f"{f[0]}({formula_to_str(f[1])})"
        if len(f) == 3:
            return f"({formula_to_str(f[1])} {f[0]} {formula_to_str(f[2])})"
    return str(f)


def tokenize_formula_string(s: str) -> List[str]:
    token_re = r"&&|\|\||->|=>|[()]|!|U|X|F|G|[A-Za-z_][A-Za-z0-9_]*"
    tokens = re.findall(token_re, s)
    if not tokens:
        raise ValueError(f"Cannot tokenize formula string: {s!r}")
    return tokens


class FormulaParser:
    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.i = 0

    def peek(self) -> Optional[str]:
        if self.i >= len(self.tokens):
            return None
        return self.tokens[self.i]

    def consume(self, expected: Optional[str] = None) -> str:
        if self.i >= len(self.tokens):
            raise ValueError("Unexpected end of formula")
        tok = self.tokens[self.i]
        if expected is not None and tok != expected:
            raise ValueError(f"Expected {expected!r}, got {tok!r}")
        self.i += 1
        return tok

    def parse(self):
        tree = self.parse_expr()
        if self.peek() is not None:
            raise ValueError(f"Unconsumed tokens: {self.tokens[self.i:]} in {self.tokens}")
        return tree

    def parse_expr(self):
        tok = self.peek()
        if tok is None:
            raise ValueError("Empty expression")

        if tok in ["X", "F", "G", "!", "not"]:
            op = self.consume()
            if self.peek() == "(":
                self.consume("(")
                child = self.parse_expr()
                self.consume(")")
            else:
                child = self.parse_expr()
            return (op, child)

        if tok == "(":
            self.consume("(")
            left = self.parse_expr()
            nxt = self.peek()
            if nxt == ")":
                self.consume(")")
                return left
            op = self.consume()
            right = self.parse_expr()
            self.consume(")")
            return (op, left, right)

        # atom / constant
        atom = self.consume()
        return atom.lower()


def formula_string_to_tree(s: str):
    return FormulaParser(tokenize_formula_string(s)).parse()


def load_formula_records(formula_json_path: Path, top_k: int = 20) -> Tuple[List[dict], List[object]]:
    with open(formula_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and "formulas" in payload:
        records = payload["formulas"]
    else:
        raise ValueError(
            f"Unsupported formula JSON format: {formula_json_path}. "
            "Expected a list of formula records or a dict with key 'formulas'."
        )

    if top_k and top_k > 0:
        records = records[:top_k]

    trees = []
    clean_records = []

    for k, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"Formula item {k} is not a dict: {item!r}")

        if "formula_tree" in item:
            tree = jsonable_to_formula(item["formula_tree"])
        elif "formula_tree_repr" in item:
            tree = ast.literal_eval(item["formula_tree_repr"])
        elif "formula" in item:
            tree = formula_string_to_tree(str(item["formula"]))
        elif "formula_str" in item:
            tree = formula_string_to_tree(str(item["formula_str"]))
        elif "content" in item:
            tree = formula_string_to_tree(str(item["content"]))
        else:
            raise ValueError(
                f"Formula item {k} has no formula_tree, formula_tree_repr, formula, formula_str, or content. "
                f"Keys={list(item.keys())}"
            )

        rec = dict(item)
        rec.setdefault("formula_id", f"T{k + 1:02d}")
        rec.setdefault("formula_str", formula_to_str(tree))
        rec["formula_tree"] = formula_to_jsonable(tree)
        rec["formula_tree_repr"] = repr(tree)
        rec.setdefault("selected_rank_by_val", rec.get("semantic_unique_rank", k + 1))

        clean_records.append(rec)
        trees.append(tree)

    return clean_records, trees


def finite_ltl_eval(tree, seq: Sequence[str], pos: int, memo: dict) -> bool:
    """
    Minimal finite-trace LTLf evaluator.

    Supported:
      atoms: strings. If seq[pos] is a list/set of propositions, atom a means a in seq[pos].
             If seq[pos] is a string, atom a means seq[pos] == a.
      constants: true, false
      unary: !, not, X, N, F, G
      binary: &&, &, and, ||, |, or, U, ->, =>, implies

    X is strong next. U is finite-trace until.
    """
    key = (repr(tree), pos)
    if key in memo:
        return memo[key]

    n = len(seq)

    if isinstance(tree, str):
        atom = tree.strip().lower()
        if atom == "true":
            ans = True
        elif atom == "false":
            ans = False
        else:
            if 0 <= pos < n:
                state = seq[pos]
                if isinstance(state, (list, tuple, set)):
                    ans = atom in {str(v).strip().lower() for v in state}
                else:
                    ans = str(state).strip().lower() == atom
            else:
                ans = False
        memo[key] = ans
        return ans

    if not isinstance(tree, tuple) or len(tree) < 2:
        raise ValueError(f"Unsupported formula tree node: {tree!r}")

    op = str(tree[0])

    if op in ["!", "not"]:
        ans = not finite_ltl_eval(tree[1], seq, pos, memo)
    elif op in ["X", "N"]:
        ans = (pos + 1 < n) and finite_ltl_eval(tree[1], seq, pos + 1, memo)
    elif op in ["F", "eventually"]:
        ans = any(finite_ltl_eval(tree[1], seq, j, memo) for j in range(pos, n))
    elif op in ["G", "always"]:
        ans = all(finite_ltl_eval(tree[1], seq, j, memo) for j in range(pos, n))
    elif op in ["&&", "&", "and"]:
        ans = finite_ltl_eval(tree[1], seq, pos, memo) and finite_ltl_eval(tree[2], seq, pos, memo)
    elif op in ["||", "|", "or"]:
        ans = finite_ltl_eval(tree[1], seq, pos, memo) or finite_ltl_eval(tree[2], seq, pos, memo)
    elif op in ["->", "=>", "implies"]:
        ans = (not finite_ltl_eval(tree[1], seq, pos, memo)) or finite_ltl_eval(tree[2], seq, pos, memo)
    elif op == "U":
        left, right = tree[1], tree[2]
        ans = False
        for j in range(pos, n):
            if finite_ltl_eval(right, seq, j, memo):
                ok_prefix = True
                for k in range(pos, j):
                    if not finite_ltl_eval(left, seq, k, memo):
                        ok_prefix = False
                        break
                if ok_prefix:
                    ans = True
                    break
    else:
        raise ValueError(f"Unsupported LTL operator: {op!r} in node {tree!r}")

    memo[key] = ans
    return ans


def truncate_sequence(seq: Sequence[str], max_len: int, mode: str = "last") -> List[str]:
    if max_len <= 0:
        return list(seq)
    if len(seq) <= max_len:
        return list(seq)
    if mode == "first":
        return list(seq[:max_len])
    if mode == "last":
        return list(seq[-max_len:])
    raise ValueError("mode must be 'first' or 'last'")


def model_check_tag(mc_window: int, mc_truncation: str) -> str:
    if int(mc_window) <= 0:
        return "fullcheck"
    return f"{mc_truncation}{int(mc_window)}check"


def compute_ltl_matrix_from_trees(
    sequences: Sequence[Sequence[str]],
    trees: Sequence[object],
    mc_window: int,
    mc_truncation: str,
    progress_every: int = 50000,
) -> Tuple[np.ndarray, float]:
    z = np.zeros((len(sequences), len(trees)), dtype=np.float32)
    t0 = time.perf_counter()

    for i, seq in enumerate(sequences):
        mc_seq = truncate_sequence(seq, max_len=mc_window, mode=mc_truncation)
        for k, tree in enumerate(trees):
            z[i, k] = float(finite_ltl_eval(tree, mc_seq, 0, memo={}))

        if progress_every > 0 and (i + 1) % progress_every == 0:
            print(f"  model checked {i + 1}/{len(sequences)} traces")

    elapsed = time.perf_counter() - t0
    print(f"Model checking finished: Z={z.shape}, time={elapsed:.2f}s")
    return z, elapsed




# ============================================================
# Automaton-state / formula-progression representation
# ============================================================

TRUE_FORMULA = "true"
FALSE_FORMULA = "false"


def normalize_formula_op(op: str) -> str:
    op = str(op)
    lower = op.lower()
    if lower in ["and", "&"]:
        return "&&"
    if lower in ["or", "|"]:
        return "||"
    if lower in ["not"]:
        return "!"
    if lower in ["n"]:
        return "X"
    if lower in ["eventually"]:
        return "F"
    if lower in ["always"]:
        return "G"
    if lower in ["implies", "=>"]:
        return "->"
    return op


def is_true_formula(f) -> bool:
    return isinstance(f, str) and f.strip().lower() == TRUE_FORMULA


def is_false_formula(f) -> bool:
    return isinstance(f, str) and f.strip().lower() == FALSE_FORMULA


def formula_sort_key(f) -> str:
    return formula_to_str(f)


def make_not(f):
    f = simplify_formula(f)
    if is_true_formula(f):
        return FALSE_FORMULA
    if is_false_formula(f):
        return TRUE_FORMULA
    if isinstance(f, tuple) and normalize_formula_op(f[0]) == "!":
        return simplify_formula(f[1])
    return ("!", f)


def make_and(a, b):
    a = simplify_formula(a)
    b = simplify_formula(b)
    if is_false_formula(a) or is_false_formula(b):
        return FALSE_FORMULA
    if is_true_formula(a):
        return b
    if is_true_formula(b):
        return a
    if formula_sort_key(a) == formula_sort_key(b):
        return a
    # Canonicalize commutative operands to reduce residual-state duplicates.
    if formula_sort_key(a) > formula_sort_key(b):
        a, b = b, a
    return ("&&", a, b)


def make_or(a, b):
    a = simplify_formula(a)
    b = simplify_formula(b)
    if is_true_formula(a) or is_true_formula(b):
        return TRUE_FORMULA
    if is_false_formula(a):
        return b
    if is_false_formula(b):
        return a
    if formula_sort_key(a) == formula_sort_key(b):
        return a
    # Canonicalize commutative operands to reduce residual-state duplicates.
    if formula_sort_key(a) > formula_sort_key(b):
        a, b = b, a
    return ("||", a, b)


def make_until(a, b):
    a = simplify_formula(a)
    b = simplify_formula(b)
    if is_true_formula(b):
        return TRUE_FORMULA
    if is_false_formula(a) and is_false_formula(b):
        return FALSE_FORMULA
    if is_false_formula(a):
        return b
    return ("U", a, b)


def simplify_formula(f):
    """Lightweight Boolean/LTL simplifier used for residual formula states."""
    if isinstance(f, str):
        atom = f.strip().lower()
        if atom in ["true", "false"]:
            return atom
        return atom
    if not isinstance(f, tuple):
        return f
    if len(f) == 2:
        op = normalize_formula_op(f[0])
        child = simplify_formula(f[1])
        if op == "!":
            return make_not(child)
        if op == "X":
            # X true/false can be simplified in a continuation semantics.
            if is_true_formula(child) or is_false_formula(child):
                return child
            return ("X", child)
        if op == "F":
            if is_true_formula(child):
                return TRUE_FORMULA
            if is_false_formula(child):
                return FALSE_FORMULA
            return ("F", child)
        if op == "G":
            if is_false_formula(child):
                return FALSE_FORMULA
            if is_true_formula(child):
                return TRUE_FORMULA
            return ("G", child)
        return (op, child)
    if len(f) == 3:
        op = normalize_formula_op(f[0])
        a = simplify_formula(f[1])
        b = simplify_formula(f[2])
        if op == "&&":
            return make_and(a, b)
        if op == "||":
            return make_or(a, b)
        if op == "->":
            return make_or(make_not(a), b)
        if op == "U":
            return make_until(a, b)
        return (op, a, b)
    return f


def state_props(state) -> set:
    if isinstance(state, (list, tuple, set)):
        return {str(v).strip().lower() for v in state if str(v).strip()}
    tok = str(state).strip().lower()
    if "_" in tok:
        return {p for p in tok.split("_") if p}
    return {tok} if tok else set()


def progress_formula_one_step(f, state):
    """
    Formula progression over one observed state.

    The residual formula is an automaton state: after reading the current prefix,
    it describes what must still hold on future continuation for the original
    formula to be satisfied. This gives a prefix-aware representation that is
    richer than final binary satisfaction.
    """
    f = simplify_formula(f)
    props = state_props(state)

    if isinstance(f, str):
        atom = f.strip().lower()
        if atom == TRUE_FORMULA:
            return TRUE_FORMULA
        if atom == FALSE_FORMULA:
            return FALSE_FORMULA
        return TRUE_FORMULA if atom in props else FALSE_FORMULA

    if not isinstance(f, tuple):
        return FALSE_FORMULA

    op = normalize_formula_op(f[0])
    if len(f) == 2:
        child = f[1]
        if op == "!":
            return make_not(progress_formula_one_step(child, state))
        if op == "X":
            # After consuming one symbol, X phi becomes phi.
            return simplify_formula(child)
        if op == "F":
            return make_or(progress_formula_one_step(child, state), ("F", simplify_formula(child)))
        if op == "G":
            return make_and(progress_formula_one_step(child, state), ("G", simplify_formula(child)))
        return simplify_formula((op, progress_formula_one_step(child, state)))

    if len(f) == 3:
        left, right = f[1], f[2]
        if op == "&&":
            return make_and(progress_formula_one_step(left, state), progress_formula_one_step(right, state))
        if op == "||":
            return make_or(progress_formula_one_step(left, state), progress_formula_one_step(right, state))
        if op == "->":
            return make_or(make_not(progress_formula_one_step(left, state)), progress_formula_one_step(right, state))
        if op == "U":
            # prog(phi U psi) = prog(psi) OR (prog(phi) AND (phi U psi))
            return make_or(
                progress_formula_one_step(right, state),
                make_and(progress_formula_one_step(left, state), make_until(left, right)),
            )
        return simplify_formula((op, progress_formula_one_step(left, state), progress_formula_one_step(right, state)))

    return f


def residual_after_prefix(tree, prefix_seq: Sequence[object]):
    residual = simplify_formula(tree)
    for state in prefix_seq:
        residual = simplify_formula(progress_formula_one_step(residual, state))
    return simplify_formula(residual)


def formula_state_key(f) -> str:
    return formula_to_str(simplify_formula(f))


def formula_size(f) -> int:
    f = simplify_formula(f)
    if isinstance(f, str):
        return 1
    if isinstance(f, tuple):
        return 1 + sum(formula_size(x) for x in f[1:])
    return 1


def formula_depth(f) -> int:
    f = simplify_formula(f)
    if isinstance(f, str):
        return 1
    if isinstance(f, tuple):
        return 1 + max([formula_depth(x) for x in f[1:]] or [0])
    return 1


def ltl_representation_tag(args) -> str:
    rep = getattr(args, "ltl_representation", "binary")
    return model_check_tag(args.mc_window, args.mc_truncation) if rep == "binary" else f"{model_check_tag(args.mc_window, args.mc_truncation)}_{rep}"


def compute_automaton_state_matrix_from_trees(
    sequences: Sequence[Sequence[object]],
    trees: Sequence[object],
    mc_window: int,
    mc_truncation: str,
    progress_every: int = 50000,
    max_states_per_formula: int = 64,
    include_binary: bool = True,
) -> Tuple[np.ndarray, float, List[str], dict]:
    """
    Build an automaton-state representation using formula progression.

    For each formula and prefix, we compute the residual formula after reading the
    prefix. The residual formula is the state of the formula-progression automaton.
    Features include:
      - optional binary finite-prefix satisfaction
      - true / false / pending residual-state indicators
      - residual formula size and depth
      - one-hot residual-state id per formula, with rare states collapsed to OTHER

    This is not the old final-satisfaction Z. It is a progress-aware prefix
    representation intended for online/prefix experiments.
    """
    n = len(sequences)
    m = len(trees)
    t0 = time.perf_counter()

    state_keys = [[None] * m for _ in range(n)]
    binary = np.zeros((n, m), dtype=np.float32)
    numeric = np.zeros((n, m, 5), dtype=np.float32)  # true, false, pending, log_size, depth
    counts = [dict() for _ in range(m)]

    for i, seq in enumerate(sequences):
        mc_seq = truncate_sequence(seq, max_len=mc_window, mode=mc_truncation)
        for k, tree in enumerate(trees):
            residual = residual_after_prefix(tree, mc_seq)
            key = formula_state_key(residual)
            state_keys[i][k] = key
            counts[k][key] = counts[k].get(key, 0) + 1

            is_true = float(is_true_formula(residual))
            is_false = float(is_false_formula(residual))
            is_pending = float((not is_true) and (not is_false))
            numeric[i, k, 0] = is_true
            numeric[i, k, 1] = is_false
            numeric[i, k, 2] = is_pending
            numeric[i, k, 3] = math.log1p(formula_size(residual))
            numeric[i, k, 4] = float(formula_depth(residual))

            if include_binary:
                binary[i, k] = float(finite_ltl_eval(tree, mc_seq, 0, memo={}))

        if progress_every > 0 and (i + 1) % progress_every == 0:
            print(f"  automaton-state processed {i + 1}/{len(sequences)} traces")

    # Build per-formula residual-state vocabularies.
    state_vocabs = []
    for k in range(m):
        items = sorted(counts[k].items(), key=lambda kv: (-kv[1], kv[0]))
        keep = [key for key, _ in items[:max_states_per_formula]]
        if len(items) > max_states_per_formula:
            keep.append("<OTHER>")
        state_vocabs.append({key: idx for idx, key in enumerate(keep)})

    blocks = []
    feature_names = []

    for k in range(m):
        fid = f"T{k + 1:02d}"
        if include_binary:
            blocks.append(binary[:, [k]])
            feature_names.append(f"{fid}_binary_prefix_sat")

        # Progress flags and residual complexity.
        blocks.append(numeric[:, k, :])
        feature_names.extend([
            f"{fid}_residual_true",
            f"{fid}_residual_false",
            f"{fid}_residual_pending",
            f"{fid}_residual_log_size",
            f"{fid}_residual_depth",
        ])

        vocab = state_vocabs[k]
        onehot = np.zeros((n, len(vocab)), dtype=np.float32)
        other_idx = vocab.get("<OTHER>", None)
        inv_names = [None] * len(vocab)
        for key, idx in vocab.items():
            inv_names[idx] = key

        for i in range(n):
            key = state_keys[i][k]
            idx = vocab.get(key, other_idx)
            if idx is not None:
                onehot[i, idx] = 1.0

        blocks.append(onehot)
        for idx, key in enumerate(inv_names):
            safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(key))[:50]
            feature_names.append(f"{fid}_state_{idx:02d}_{safe}")

    z = np.concatenate(blocks, axis=1).astype(np.float32) if blocks else np.zeros((n, 0), dtype=np.float32)
    elapsed = time.perf_counter() - t0
    metadata = {
        "representation": "automaton_state_progression",
        "include_binary": bool(include_binary),
        "max_states_per_formula": int(max_states_per_formula),
        "state_vocabs": state_vocabs,
        "feature_names": feature_names,
    }
    print(f"Automaton-state representation finished: Z={z.shape}, time={elapsed:.2f}s")
    return z, elapsed, feature_names, metadata


def compute_ltl_representation_from_trees(
    sequences: Sequence[Sequence[object]],
    trees: Sequence[object],
    args,
) -> Tuple[np.ndarray, float, List[str], dict]:
    rep = getattr(args, "ltl_representation", "binary")
    if rep == "binary":
        z, elapsed = compute_ltl_matrix_from_trees(
            sequences=sequences,
            trees=trees,
            mc_window=args.mc_window,
            mc_truncation=args.mc_truncation,
            progress_every=args.progress_every,
        )
        feature_names = [f"T{k + 1:02d}" for k in range(z.shape[1])]
        return z, elapsed, feature_names, {"representation": "binary"}

    if rep in ["automaton", "automaton_no_binary"]:
        include_binary = (rep == "automaton")
        return compute_automaton_state_matrix_from_trees(
            sequences=sequences,
            trees=trees,
            mc_window=args.mc_window,
            mc_truncation=args.mc_truncation,
            progress_every=args.progress_every,
            max_states_per_formula=args.max_states_per_formula,
            include_binary=include_binary,
        )

    raise ValueError(f"Unsupported --ltl_representation: {rep}")

# ============================================================
# Data parsing
# ============================================================

DEFAULT_ACTION_TOKEN_MAP = {
    "1": "b", "2": "d", "3": "a", "4": "r", "5": "p",
    "b": "b", "d": "d", "a": "a", "r": "r", "p": "p",
    "browse": "b", "page view": "b", "page_view": "b",
    "detail": "d", "view_detail": "d", "product_detail": "d",
    "add": "a", "add_to_cart": "a",
    "remove": "r", "abandon": "r",
    "purchase": "p",
}


def parse_raw_sequence(x) -> List[object]:
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []

    s = str(x).strip()
    if not s:
        return []

    s2 = s.replace("<", "[").replace(">", "]")
    try:
        obj = ast.literal_eval(s2)
        if isinstance(obj, (list, tuple, np.ndarray)):
            return list(obj)
    except Exception:
        pass

    s2 = s.strip("[](){}<>")
    parts = [p.strip() for p in re.split(r"[,\s;]+", s2) if p.strip()]
    return parts


def normalize_token(token, action_token_map: Dict[str, str]) -> str:
    t = str(token).strip().strip("'\"").lower()
    return action_token_map.get(t, t)


def parse_behavior_sequence(
    x,
    action_token_map: Dict[str, str],
    remove_purchase_tokens: bool = True,
    purchase_tokens: Optional[set] = None,
) -> List[str]:
    if purchase_tokens is None:
        purchase_tokens = {"p", "purchase", "5"}

    raw = parse_raw_sequence(x)
    out = []

    for tok in raw:
        norm = normalize_token(tok, action_token_map)
        if remove_purchase_tokens and norm in purchase_tokens:
            continue
        if norm:
            out.append(norm)

    return out


def sequence_length_stats(sequences: Sequence[Sequence[str]], seed, split_name: str) -> dict:
    lengths = np.array([len(s) for s in sequences], dtype=float)
    if len(lengths) == 0:
        return {
            "seed": seed, "split": split_name, "n": 0,
            "min_len": np.nan, "mean_len": np.nan, "median_len": np.nan,
            "p90_len": np.nan, "p95_len": np.nan, "p99_len": np.nan, "max_len": np.nan,
        }
    return {
        "seed": seed,
        "split": split_name,
        "n": int(len(lengths)),
        "min_len": int(lengths.min()),
        "mean_len": float(lengths.mean()),
        "median_len": float(np.median(lengths)),
        "p90_len": float(np.percentile(lengths, 90)),
        "p95_len": float(np.percentile(lengths, 95)),
        "p99_len": float(np.percentile(lengths, 99)),
        "max_len": int(lengths.max()),
    }




def parse_numeric_sequence(x) -> List[float]:
    if isinstance(x, list):
        vals = x
    elif isinstance(x, tuple):
        vals = list(x)
    elif isinstance(x, np.ndarray):
        vals = x.tolist()
    elif pd.isna(x):
        vals = []
    else:
        s = str(x).strip()
        if not s:
            vals = []
        else:
            s2 = s.replace("<", "[").replace(">", "]")
            try:
                obj = ast.literal_eval(s2)
                if isinstance(obj, (list, tuple, np.ndarray)):
                    vals = list(obj)
                else:
                    vals = [obj]
            except Exception:
                s2 = s.strip("[](){}<>")
                vals = [p.strip() for p in re.split(r"[,\s;]+", s2) if p.strip()]
    out = []
    for v in vals:
        try:
            fv = float(v)
            if np.isfinite(fv):
                out.append(fv)
            else:
                out.append(0.0)
        except Exception:
            out.append(0.0)
    return out


def normalize_dwell_bucket_token(x: str) -> str:
    t = str(x).strip().strip("'\"").lower()
    mapping = {
        "short": "ds", "s": "ds", "low": "ds", "small": "ds",
        "medium": "dm", "med": "dm", "m": "dm", "middle": "dm",
        "long": "dh", "high": "dh", "h": "dh", "large": "dh",
        "ds": "ds", "dm": "dm", "dh": "dh", "dl": "dl",
    }
    return mapping.get(t, t)


def parse_logic_trace_cell(x) -> List[List[str]]:
    """Parse a TLTLf-style trace cell: [["b","dm"], ["d","dh"]]."""
    if isinstance(x, list):
        obj = x
    elif isinstance(x, tuple):
        obj = list(x)
    elif isinstance(x, np.ndarray):
        obj = x.tolist()
    elif pd.isna(x):
        return []
    else:
        s = str(x).strip()
        if not s:
            return []
        try:
            obj = json.loads(s)
        except Exception:
            s2 = s.replace("<", "[").replace(">", "]")
            try:
                obj = ast.literal_eval(s2)
            except Exception:
                # Fallback: treat as a flat action/combined-token sequence.
                obj = parse_raw_sequence(s)

    trace = []
    for state in obj:
        if isinstance(state, (list, tuple, set)):
            props = [str(v).strip().strip("'\"").lower() for v in state if str(v).strip()]
        else:
            tok = str(state).strip().strip("'\"").lower()
            if "_" in tok:
                props = [p for p in tok.split("_") if p]
            else:
                props = [tok] if tok else []
        if props:
            trace.append(props)
    return trace


def build_logic_trace_from_action_and_dwell_bucket(action_seq, dwell_bucket_seq, remove_purchase_tokens=True) -> List[List[str]]:
    actions = [normalize_token(x, DEFAULT_ACTION_TOKEN_MAP) for x in action_seq]
    dwells = [normalize_dwell_bucket_token(x) for x in dwell_bucket_seq]
    n = min(len(actions), len(dwells))
    trace = []
    for a, d in zip(actions[:n], dwells[:n]):
        if remove_purchase_tokens and a in {"p", "purchase", "5"}:
            continue
        if a and d:
            trace.append([a, d])
    return trace


def build_logic_trace_for_row(row, args, action_seq) -> List[List[str]]:
    # Preferred: explicit TLTLf trace/logic_trace column created by prepare_behavior_dwell_tltlf_json.py.
    if args.logic_trace_col and args.logic_trace_col in row.index and not pd.isna(row[args.logic_trace_col]):
        trace = parse_logic_trace_cell(row[args.logic_trace_col])
        if trace:
            return trace

    # Next: combine action sequence with dwell buckets.
    if args.dwell_bucket_col and args.dwell_bucket_col in row.index and not pd.isna(row[args.dwell_bucket_col]):
        dwell_bucket_seq = parse_raw_sequence(row[args.dwell_bucket_col])
        trace = build_logic_trace_from_action_and_dwell_bucket(
            action_seq, dwell_bucket_seq, remove_purchase_tokens=args.remove_purchase_tokens
        )
        if trace:
            return trace

    # Next: if the sequence column already contains combined tokens like d_dh, split them into state propositions.
    if len(action_seq) > 0 and any("_" in str(tok) for tok in action_seq):
        return [[p for p in str(tok).lower().split("_") if p] for tok in action_seq]

    # Fallback: action-only propositions.
    return [[str(tok).lower()] for tok in action_seq]


def align_numeric_to_action(dwell_seq: Sequence[float], action_seq: Sequence[str], remove_purchase_tokens=True) -> List[float]:
    n = min(len(dwell_seq), len(action_seq))
    out = []
    for d, a in zip(list(dwell_seq)[:n], list(action_seq)[:n]):
        if remove_purchase_tokens and str(a).strip().lower() in {"p", "purchase", "5"}:
            continue
        out.append(float(d))
    return out
def load_seed_split_data(args, seed: int):
    seed_dir = Path(args.aligned_splits_dir) / f"seed_{seed}"
    frames = []
    length_rows = []

    for split in ["train", "val", "test"]:
        path = seed_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing split CSV: {path}")
        df = pd.read_csv(path).copy()

        for col in [args.seq_col, args.label_col, args.session_col]:
            if col not in df.columns:
                raise KeyError(f"Missing column {col} in {path}. Available columns: {list(df.columns)}")
        if args.require_dwell_seconds and args.dwell_seconds_col not in df.columns:
            raise KeyError(f"Missing dwell seconds column {args.dwell_seconds_col} in {path}. Available columns: {list(df.columns)}")

        action_sequences = []
        logic_traces = []
        dwell_seconds_sequences = []

        for _, row in df.iterrows():
            action_seq = parse_behavior_sequence(
                row[args.seq_col],
                action_token_map=DEFAULT_ACTION_TOKEN_MAP,
                remove_purchase_tokens=args.remove_purchase_tokens,
            )
            action_sequences.append(action_seq)

            logic_traces.append(build_logic_trace_for_row(row, args, action_seq))

            if args.dwell_seconds_col in df.columns:
                raw_action_seq_for_alignment = parse_raw_sequence(row[args.seq_col])
                raw_dwell_seq = parse_numeric_sequence(row[args.dwell_seconds_col])
                dwell_seconds_sequences.append(
                    align_numeric_to_action(raw_dwell_seq, raw_action_seq_for_alignment, remove_purchase_tokens=args.remove_purchase_tokens)
                )
            else:
                dwell_seconds_sequences.append([0.0] * len(action_seq))

        df["_split"] = split
        df["_parsed_sequence"] = action_sequences
        df["_logic_trace"] = logic_traces
        df["_dwell_seconds_sequence"] = dwell_seconds_sequences
        frames.append(df)
        length_rows.append(sequence_length_stats(action_sequences, seed, split))

    full_df = pd.concat(frames, ignore_index=True)
    length_rows.append(sequence_length_stats(full_df["_parsed_sequence"].tolist(), seed, "all"))

    return full_df, length_rows

def build_action_vocab(sequences: Sequence[Sequence[str]], preferred_order=("b", "d", "a", "r")) -> Dict[str, int]:
    tokens = sorted({t for seq in sequences for t in seq})
    ordered = []
    for t in preferred_order:
        if t in tokens:
            ordered.append(t)
    for t in tokens:
        if t not in ordered:
            ordered.append(t)
    return {tok: i + 1 for i, tok in enumerate(ordered)}


def encode_raw_sequence(seq: Sequence[str], action_vocab: Dict[str, int], max_len: int, truncation: str) -> np.ndarray:
    sub = truncate_sequence(seq, max_len=max_len, mode=truncation)
    arr = np.zeros(max_len, dtype=np.int64)
    for i, tok in enumerate(sub):
        arr[i] = action_vocab.get(tok, 0)
    return arr


def build_raw_matrix(sequences: Sequence[Sequence[str]], action_vocab: Dict[str, int], max_len: int, truncation: str) -> np.ndarray:
    return np.stack([encode_raw_sequence(seq, action_vocab, max_len, truncation) for seq in sequences])



def encode_dwell_sequence(seq: Sequence[float], max_len: int, truncation: str, mean: float, std: float, log1p: bool = True) -> np.ndarray:
    sub = truncate_sequence(seq, max_len=max_len, mode=truncation)
    arr = np.zeros(max_len, dtype=np.float32)
    for i, val in enumerate(sub):
        v = float(val)
        if log1p:
            v = np.log1p(max(v, 0.0))
        arr[i] = (v - mean) / max(std, 1e-8)
    return arr


def fit_dwell_scaler(sequences: Sequence[Sequence[float]], max_len: int, truncation: str, log1p: bool = True) -> Tuple[float, float]:
    vals = []
    for seq in sequences:
        sub = truncate_sequence(seq, max_len=max_len, mode=truncation)
        for v in sub:
            fv = float(v)
            if log1p:
                fv = np.log1p(max(fv, 0.0))
            vals.append(fv)
    if len(vals) == 0:
        return 0.0, 1.0
    vals = np.asarray(vals, dtype=np.float32)
    return float(vals.mean()), float(vals.std() + 1e-8)


def build_dwell_matrix(sequences: Sequence[Sequence[float]], max_len: int, truncation: str, mean: float, std: float, log1p: bool = True) -> np.ndarray:
    return np.stack([encode_dwell_sequence(seq, max_len, truncation, mean, std, log1p) for seq in sequences]).astype(np.float32)


def formula_path_for_seed(args, seed: int) -> Path:
    if args.formula_json:
        return Path(args.formula_json)

    if args.formula_mode == "fixed":
        seed_for_formula = args.formula_source_seed
    else:
        seed_for_formula = seed

    return Path(args.formula_json_pattern.format(
        aligned_splits_dir=str(Path(args.aligned_splits_dir)),
        seed=seed_for_formula,
        window=args.formula_window,
    ))


# ============================================================
# Metrics and uncertainty
# ============================================================

def safe_auc(y_true, y_prob):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return np.nan


def safe_auprc(y_true, y_prob):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return np.nan


def safe_logloss(y_true, y_prob):
    try:
        return float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7), labels=[0, 1]))
    except Exception:
        return np.nan


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob <= hi) if i == n_bins - 1 else (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(acc - conf)
    return float(ece)


def choose_threshold(y_val, p_val, metric="f1"):
    candidates = np.linspace(0.05, 0.95, 91)
    best_t, best_score = 0.5, -1.0
    for t in candidates:
        pred = (p_val >= t).astype(int)
        if metric == "mcc":
            score = matthews_corrcoef(y_val, pred)
        else:
            score = f1_score(y_val, pred, zero_division=0)
        if score > best_score:
            best_score = float(score)
            best_t = float(t)
    return best_t, best_score


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)
    npv = tn / (tn + fn + 1e-8)
    fpr = fp / (fp + tn + 1e-8)
    fnr = fn / (fn + tp + 1e-8)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "npv": float(npv),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "auc": safe_auc(y_true, y_prob),
        "auprc": safe_auprc(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "logloss": safe_logloss(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=10),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def ci95(series):
    x = pd.to_numeric(series, errors="coerce").dropna().values
    n = len(x)
    if n <= 1:
        return np.nan
    sem = np.std(x, ddof=1) / math.sqrt(n)
    try:
        from scipy.stats import t
        mult = t.ppf(0.975, df=n - 1)
    except Exception:
        mult = 1.96
    return float(mult * sem)


# ============================================================
# PyTorch datasets and models
# ============================================================

class RawDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.as_tensor(x, dtype=torch.long)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class ZDataset(Dataset):
    def __init__(self, z, y):
        self.z = torch.as_tensor(z, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.z[idx], self.y[idx]


class FusionDataset(Dataset):
    def __init__(self, raw_x, z, y):
        self.raw_x = torch.as_tensor(raw_x, dtype=torch.long)
        self.z = torch.as_tensor(z, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.raw_x[idx], self.z[idx], self.y[idx]



class RawDwellDataset(Dataset):
    def __init__(self, raw_x, dwell_x, y):
        self.raw_x = torch.as_tensor(raw_x, dtype=torch.long)
        self.dwell_x = torch.as_tensor(dwell_x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.raw_x[idx], self.dwell_x[idx], self.y[idx]


class FusionDwellDataset(Dataset):
    def __init__(self, raw_x, dwell_x, z, y):
        self.raw_x = torch.as_tensor(raw_x, dtype=torch.long)
        self.dwell_x = torch.as_tensor(dwell_x, dtype=torch.float32)
        self.z = torch.as_tensor(z, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.raw_x[idx], self.dwell_x[idx], self.z[idx], self.y[idx]


class TransformerSequenceClassifier(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, d_model=64, n_heads=4, n_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
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

    def forward(self, x):
        bsz, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, seq_len)
        mask = x.eq(0)
        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.encoder(h, src_key_padding_mask=mask)
        h = self.norm(h)
        valid = (~mask).float().unsqueeze(-1)
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.head(pooled).squeeze(-1)


class TransformerDwellSequenceClassifier(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, d_model=64, n_heads=4, n_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dwell_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
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

    def forward(self, raw_x, dwell_x):
        bsz, seq_len = raw_x.shape
        pos = torch.arange(seq_len, device=raw_x.device).unsqueeze(0).expand(bsz, seq_len)
        mask = raw_x.eq(0)
        h = self.token_emb(raw_x) + self.pos_emb(pos) + self.dwell_proj(dwell_x.unsqueeze(-1))
        h = self.input_norm(h)
        h = self.encoder(h, src_key_padding_mask=mask)
        h = self.norm(h)
        valid = (~mask).float().unsqueeze(-1)
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.head(pooled).squeeze(-1)


class LateFusionDwellTransformerLTLClassifier(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, z_dim: int, d_model=64, n_heads=4, n_layers=2,
                 dim_feedforward=128, ltl_hidden_dim=64, fusion_hidden_dim=128, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dwell_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
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
        self.raw_norm = nn.LayerNorm(d_model)
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

    def forward(self, raw_x, dwell_x, z):
        bsz, seq_len = raw_x.shape
        pos = torch.arange(seq_len, device=raw_x.device).unsqueeze(0).expand(bsz, seq_len)
        mask = raw_x.eq(0)
        h = self.token_emb(raw_x) + self.pos_emb(pos) + self.dwell_proj(dwell_x.unsqueeze(-1))
        h = self.input_norm(h)
        h = self.encoder(h, src_key_padding_mask=mask)
        h = self.raw_norm(h)
        valid = (~mask).float().unsqueeze(-1)
        h_raw = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        h_ltl = self.ltl_encoder(z)
        return self.head(torch.cat([h_raw, h_ltl], dim=1)).squeeze(-1)


class MLPBinaryClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LateFusionTransformerLTLClassifier(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, z_dim: int, d_model=64, n_heads=4, n_layers=2,
                 dim_feedforward=128, ltl_hidden_dim=64, fusion_hidden_dim=128, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
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
        self.raw_norm = nn.LayerNorm(d_model)
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

    def forward(self, raw_x, z):
        bsz, seq_len = raw_x.shape
        pos = torch.arange(seq_len, device=raw_x.device).unsqueeze(0).expand(bsz, seq_len)
        mask = raw_x.eq(0)
        h = self.token_emb(raw_x) + self.pos_emb(pos)
        h = self.encoder(h, src_key_padding_mask=mask)
        h = self.raw_norm(h)
        valid = (~mask).float().unsqueeze(-1)
        h_raw = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        h_ltl = self.ltl_encoder(z)
        return self.head(torch.cat([h_raw, h_ltl], dim=1)).squeeze(-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def predict_raw_or_z(model, loader, device):
    model.eval()
    ys, probs = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        ys.append(yb.numpy())
        probs.append(prob)
    return np.concatenate(ys), np.concatenate(probs)


@torch.no_grad()
def predict_fusion(model, loader, device):
    model.eval()
    ys, probs = [], []
    for raw_xb, zb, yb in loader:
        raw_xb = raw_xb.to(device)
        zb = zb.to(device)
        logits = model(raw_xb, zb)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        ys.append(yb.numpy())
        probs.append(prob)
    return np.concatenate(ys), np.concatenate(probs)




@torch.no_grad()
def predict_raw_dwell(model, loader, device):
    model.eval()
    ys, probs = [], []
    for raw_xb, dwell_xb, yb in loader:
        raw_xb = raw_xb.to(device)
        dwell_xb = dwell_xb.to(device)
        logits = model(raw_xb, dwell_xb)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        ys.append(yb.numpy())
        probs.append(prob)
    return np.concatenate(ys), np.concatenate(probs)


@torch.no_grad()
def predict_fusion_dwell(model, loader, device):
    model.eval()
    ys, probs = [], []
    for raw_xb, dwell_xb, zb, yb in loader:
        raw_xb = raw_xb.to(device)
        dwell_xb = dwell_xb.to(device)
        zb = zb.to(device)
        logits = model(raw_xb, dwell_xb, zb)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        ys.append(yb.numpy())
        probs.append(prob)
    return np.concatenate(ys), np.concatenate(probs)

def train_torch_model(model, train_loader, val_loader, test_loader, y_train, args, mode: str):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = model.to(device)

    pos = float(np.sum(y_train))
    neg = float(len(y_train) - pos)
    pos_weight_value = neg / max(pos, 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val_score = -1.0
    best_epoch = -1
    patience_counter = 0

    t_train = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_n = 0.0, 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            if mode == "fusion":
                raw_xb, zb, yb = batch
                raw_xb = raw_xb.to(device)
                zb = zb.to(device)
                yb = yb.to(device)
                logits = model(raw_xb, zb)
            elif mode == "raw_dwell":
                raw_xb, dwell_xb, yb = batch
                raw_xb = raw_xb.to(device)
                dwell_xb = dwell_xb.to(device)
                yb = yb.to(device)
                logits = model(raw_xb, dwell_xb)
            elif mode == "fusion_dwell":
                raw_xb, dwell_xb, zb, yb = batch
                raw_xb = raw_xb.to(device)
                dwell_xb = dwell_xb.to(device)
                zb = zb.to(device)
                yb = yb.to(device)
                logits = model(raw_xb, dwell_xb, zb)
            else:
                xb, yb = batch
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)

            loss = criterion(logits, yb)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += float(loss.item()) * len(yb)
            total_n += len(yb)

        if mode == "fusion":
            y_val, p_val = predict_fusion(model, val_loader, device)
        elif mode == "raw_dwell":
            y_val, p_val = predict_raw_dwell(model, val_loader, device)
        elif mode == "fusion_dwell":
            y_val, p_val = predict_fusion_dwell(model, val_loader, device)
        else:
            y_val, p_val = predict_raw_or_z(model, val_loader, device)

        val_metrics = compute_metrics(y_val, p_val, threshold=0.5)
        val_score = val_metrics["f1"] if args.early_stop_metric == "f1" else val_metrics.get(args.early_stop_metric, val_metrics["f1"])

        if val_score > best_val_score:
            best_val_score = float(val_score)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if args.verbose:
            avg_loss = total_loss / max(total_n, 1)
            print(f"epoch={epoch:03d} loss={avg_loss:.4f} val_f1={val_metrics['f1']:.4f} val_auc={val_metrics['auc']:.4f}")

        if patience_counter >= args.patience:
            break

    train_time_sec = time.perf_counter() - t_train

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    # Choose threshold on validation set after best checkpoint is loaded.
    if mode == "fusion":
        y_val, p_val = predict_fusion(model, val_loader, device)
    elif mode == "raw_dwell":
        y_val, p_val = predict_raw_dwell(model, val_loader, device)
    elif mode == "fusion_dwell":
        y_val, p_val = predict_fusion_dwell(model, val_loader, device)
    else:
        y_val, p_val = predict_raw_or_z(model, val_loader, device)

    threshold, val_threshold_score = choose_threshold(y_val, p_val, metric=args.threshold_metric)

    t_infer = time.perf_counter()
    if mode == "fusion":
        y_test, p_test = predict_fusion(model, test_loader, device)
    elif mode == "raw_dwell":
        y_test, p_test = predict_raw_dwell(model, test_loader, device)
    elif mode == "fusion_dwell":
        y_test, p_test = predict_fusion_dwell(model, test_loader, device)
    else:
        y_test, p_test = predict_raw_or_z(model, test_loader, device)
    infer_time_sec = time.perf_counter() - t_infer

    metrics = compute_metrics(y_test, p_test, threshold=threshold)
    metrics.update({
        "best_epoch": int(best_epoch),
        "epochs_ran": int(best_epoch if best_epoch > 0 else args.epochs),
        "best_val_score": float(best_val_score),
        "val_threshold_score": float(val_threshold_score),
        "train_time_sec": float(train_time_sec),
        "infer_time_sec": float(infer_time_sec),
        "n_params": int(count_parameters(model)),
        "device": str(device),
        "pos_weight": float(pos_weight_value),
    })
    return metrics


# ============================================================
# Saving Z artifacts and tables
# ============================================================

def compute_rule_stats(z, y, formula_items):
    rows = []
    base_rate = float(np.mean(y)) if len(y) else np.nan
    for k in range(z.shape[1]):
        sat = z[:, k] >= 0.5
        n_sat = int(sat.sum())
        n_unsat = int((~sat).sum())
        y_sat = y[sat]
        y_unsat = y[~sat]
        purchase_rate_sat = float(y_sat.mean()) if n_sat else np.nan
        purchase_rate_unsat = float(y_unsat.mean()) if n_unsat else np.nan
        recall_if_sat = float(((sat) & (y == 1)).sum() / max(1, int((y == 1).sum())))
        lift = purchase_rate_sat / base_rate if base_rate and not np.isnan(base_rate) and n_sat else np.nan
        item = formula_items[k] if k < len(formula_items) else {}
        rows.append({
            "feature_idx": k,
            "formula_id": item.get("formula_id", f"T{k + 1:02d}"),
            "formula_str": item.get("formula_str", ""),
            "selected_rank_by_val": item.get("selected_rank_by_val", k + 1),
            "support": n_sat,
            "support_rate": float(sat.mean()) if len(sat) else np.nan,
            "purchase_rate_if_satisfied": purchase_rate_sat,
            "purchase_rate_if_not_satisfied": purchase_rate_unsat,
            "precision_if_satisfied": purchase_rate_sat,
            "recall_if_satisfied": recall_if_sat,
            "lift_vs_base_rate": lift,
            "base_purchase_rate": base_rate,
            "pos_satisfaction_rate": float(z[y == 1, k].mean()) if np.any(y == 1) else np.nan,
            "neg_satisfaction_rate": float(z[y == 0, k].mean()) if np.any(y == 0) else np.nan,
        })
    return pd.DataFrame(rows)


def save_z_artifacts(seed_out_dir, file_stem, z, session_ids, labels, split_labels, formula_items,
                     model_checking_time_sec, args, formula_json_path, feature_names=None, representation_meta=None):
    seed_out_dir.mkdir(parents=True, exist_ok=True)
    formula_id = np.array([item.get("formula_id", f"T{k + 1:02d}") for k, item in enumerate(formula_items)], dtype=object)
    formula_str = np.array([item.get("formula_str", "") for item in formula_items], dtype=object)
    formula_tree_repr = np.array([item.get("formula_tree_repr", "") for item in formula_items], dtype=object)

    npz_path = seed_out_dir / f"{file_stem}.npz"
    csv_path = seed_out_dir / f"{file_stem}.csv"

    np.savez_compressed(
        npz_path,
        Z=z.astype(np.float32),
        session_id_hash=np.asarray(session_ids).astype(str),
        y=labels.astype(np.int64),
        split=np.asarray(split_labels).astype(object),
        formula_id=formula_id,
        formula_str=formula_str,
        formula_tree_repr=formula_tree_repr,
        model_checking_time_sec=np.array([model_checking_time_sec], dtype=np.float64),
        mc_window=np.array([args.mc_window], dtype=np.int64),
        mc_truncation=np.array([args.mc_truncation], dtype=object),
        formula_json=np.array([str(formula_json_path)], dtype=object),
        action_token_map=np.array([json.dumps(DEFAULT_ACTION_TOKEN_MAP, ensure_ascii=False)], dtype=object),
        remove_purchase_tokens=np.array([bool(args.remove_purchase_tokens)], dtype=bool),
        ltl_representation=np.array([getattr(args, "ltl_representation", "binary")], dtype=object),
        feature_names=np.asarray(feature_names if feature_names is not None else [f"T{k + 1:02d}" for k in range(z.shape[1])], dtype=object),
    )

    if feature_names is None or len(feature_names) != z.shape[1]:
        feature_names = [f"T{k + 1:02d}" for k in range(z.shape[1])]
    z_df = pd.DataFrame(z.astype(np.float32), columns=feature_names)
    z_df.insert(0, "session_id_hash", np.asarray(session_ids).astype(str))
    z_df.insert(1, "split", np.asarray(split_labels).astype(str))
    z_df.insert(2, args.label_col, labels.astype(int))
    z_df.to_csv(csv_path, index=False)

    with open(seed_out_dir / f"{file_stem}_formulas_used.json", "w", encoding="utf-8") as f:
        json.dump(list(formula_items), f, ensure_ascii=False, indent=2)

    if getattr(args, "ltl_representation", "binary") == "binary" and z.shape[1] == len(formula_items):
        rule_stats = compute_rule_stats(z, labels, formula_items)
        rule_stats.to_csv(seed_out_dir / f"{file_stem}_rule_stats.csv", index=False)
    else:
        feature_stats = pd.DataFrame({
            "feature_idx": np.arange(z.shape[1]),
            "feature_name": feature_names,
            "mean": z.mean(axis=0),
            "std": z.std(axis=0),
            "min": z.min(axis=0),
            "max": z.max(axis=0),
        })
        feature_stats.to_csv(seed_out_dir / f"{file_stem}_feature_stats.csv", index=False)

    with open(seed_out_dir / f"{file_stem}_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "Z_shape": list(z.shape),
            "model_checking_time_sec": float(model_checking_time_sec),
            "mc_window": int(args.mc_window),
            "mc_tag": model_check_tag(args.mc_window, args.mc_truncation),
            "ltl_representation": args.ltl_representation,
            "ltl_representation_tag": ltl_representation_tag(args),
            "mc_truncation": args.mc_truncation,
            "formula_json": str(formula_json_path),
            "remove_purchase_tokens": bool(args.remove_purchase_tokens),
            "ltl_representation": getattr(args, "ltl_representation", "binary"),
            "feature_names": feature_names,
            "representation_meta": representation_meta or {},
            "npz_path": str(npz_path),
            "csv_path": str(csv_path),
        }, f, ensure_ascii=False, indent=2)

    if getattr(args, "ltl_representation", "binary") == "binary":
        print("LTL matrix positive rates:")
        for k in range(z.shape[1]):
            formula = formula_items[k].get("formula_str", "") if k < len(formula_items) else ""
            print(f"  T{k + 1:02d}: {z[:, k].mean():.4f}  {formula[:90]}")
    else:
        print("Automaton-state feature means, first 20 features:")
        for k, name in enumerate(feature_names[:20]):
            print(f"  {k:03d}: {z[:, k].mean():.4f}  {name}")

    return npz_path


def make_uncertainty_summary(df):
    group_cols = ["feature_pack", "experiment_type", "config_name", "model", "length_label", "mc_tag"]
    metric_cols = [
        "accuracy", "balanced_accuracy", "precision", "recall", "sensitivity", "specificity", "npv", "fpr", "fnr",
        "f1", "f2", "mcc", "cohen_kappa", "auc", "auprc", "brier", "logloss", "ece", "threshold",
        "best_epoch", "epochs_ran", "n_params", "raw_sequence_length_used", "original_max_sequence_length", "original_mean_sequence_length",
        "train_time_sec", "infer_time_sec", "model_checking_time_sec", "total_train_infer_time_sec",
        "total_feature_time_plus_train_sec", "total_pipeline_cost_sec",
        "f1_per_min", "mcc_per_min", "auc_per_min", "f1_over_log_cost", "mcc_over_log_cost", "auc_over_log_cost",
    ]
    rows = []
    ok = df[df["status"] == "ok"].copy() if "status" in df.columns else df.copy()
    for keys, g in ok.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_runs"] = len(g)
        for col in metric_cols:
            if col not in g.columns:
                continue
            row[f"{col}_mean"] = pd.to_numeric(g[col], errors="coerce").mean()
            row[f"{col}_std"] = pd.to_numeric(g[col], errors="coerce").std(ddof=1)
            row[f"{col}_ci95"] = ci95(g[col])
        rows.append(row)
    return pd.DataFrame(rows)


def add_cost_adjusted(summary):
    df = summary.copy()
    eps = 1e-8
    if "total_pipeline_cost_sec_mean" in df.columns:
        cost_min = df["total_pipeline_cost_sec_mean"] / 60.0 + eps
        log_cost = np.log1p(df["total_pipeline_cost_sec_mean"])
        for metric in ["f1", "mcc", "auc", "auprc", "balanced_accuracy"]:
            col = f"{metric}_mean"
            if col in df.columns:
                df[f"{metric}_per_min"] = df[col] / cost_min
                df[f"{metric}_over_log_cost"] = df[col] / (log_cost + eps)
    if {"f1_mean", "mcc_mean", "auc_mean", "total_pipeline_cost_sec_mean"}.issubset(df.columns):
        perf = 0.4 * df["f1_mean"] + 0.3 * df["mcc_mean"] + 0.3 * df["auc_mean"]
        perf_norm = (perf - perf.min()) / (perf.max() - perf.min() + eps)
        cost = df["total_pipeline_cost_sec_mean"]
        cost_norm = (cost - cost.min()) / (cost.max() - cost.min() + eps)
        df["normalized_perf_score"] = perf
        df["accuracy_efficiency_score"] = perf_norm - 0.25 * cost_norm
    return df


def add_pareto_flag(summary, performance_col="f1_mean", cost_col="total_pipeline_cost_sec_mean"):
    df = summary.copy()
    if performance_col not in df.columns or cost_col not in df.columns:
        df["pareto_f1_cost"] = False
        return df
    values = df[[performance_col, cost_col]].values
    pareto = []
    for i, (perf_i, cost_i) in enumerate(values):
        dominated = False
        for j, (perf_j, cost_j) in enumerate(values):
            if i == j:
                continue
            if (perf_j >= perf_i) and (cost_j <= cost_i) and ((perf_j > perf_i) or (cost_j < cost_i)):
                dominated = True
                break
        pareto.append(not dominated)
    df["pareto_f1_cost"] = pareto
    return df


METRIC_DIRECTION = {
    "accuracy": "up", "balanced_accuracy": "up", "precision": "up", "recall": "up", "specificity": "up",
    "f1": "up", "f2": "up", "mcc": "up", "cohen_kappa": "up", "auc": "up", "auprc": "up",
    "brier": "down", "logloss": "down", "ece": "down",
    "train_time_sec": "down", "infer_time_sec": "down", "model_checking_time_sec": "down", "total_pipeline_cost_sec": "down",
    "f1_per_min": "up", "mcc_per_min": "up", "auc_per_min": "up", "f1_over_log_cost": "up",
}

DISPLAY_METRIC_NAMES = {
    "accuracy": "Acc.", "balanced_accuracy": "Bal. Acc.", "precision": "Prec.", "recall": "Recall",
    "specificity": "Spec.", "f1": "F1", "f2": "F2", "mcc": "MCC", "cohen_kappa": "Kappa",
    "auc": "AUC", "auprc": "AUPRC", "brier": "Brier", "logloss": "LogLoss", "ece": "ECE",
    "train_time_sec": "Train s", "infer_time_sec": "Infer s", "model_checking_time_sec": "MC s", "total_pipeline_cost_sec": "Total s",
    "f1_per_min": "F1/min", "mcc_per_min": "MCC/min", "auc_per_min": "AUC/min", "f1_over_log_cost": "F1/logCost",
}


def latex_escape(s):
    s = str(s)
    for k, v in {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_"}.items():
        s = s.replace(k, v)
    return s


def fmt_mean_std(mean, std, metric, bold=False):
    if pd.isna(mean):
        return "--"
    decimals = 3 if metric in ["brier", "logloss", "ece"] else 2
    if "time_sec" in metric or metric.endswith("_sec"):
        if mean >= 100:
            decimals = 0
        elif mean >= 10:
            decimals = 1
        else:
            decimals = 2
    if pd.isna(std):
        text = f"{mean:.{decimals}f}"
    else:
        text = f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"
    if bold:
        return r"\textbf{" + text + "}"
    return text


def table_task_from_length(length_label):
    if str(length_label) == "full":
        return "Full raw"
    if str(length_label) == "verification_only":
        return "Verification"
    return f"Length {length_label}"


def prepare_table_rows(summary, raw_lengths):
    rows = []
    prefix_lengths = [str(x) for x in raw_lengths]
    table_order = prefix_lengths + ["full"]
    config_order = {"raw": 0, "raw_dwell": 1, "verification_only": 2, "raw_plus_ltl": 3, "raw_dwell_plus_ltl": 4}

    for _, r in summary.iterrows():
        exp = str(r.get("experiment_type", ""))
        length = str(r.get("length_label", ""))
        if exp == "verification_only":
            for L in prefix_lengths:
                rr = r.copy()
                rr["table_length"] = L
                rr["table_task"] = f"Length {L}"
                rr["table_config"] = "Symbolic"
                rr["table_config_key"] = "verification_only"
                rows.append(rr)
        elif length in table_order:
            rr = r.copy()
            rr["table_length"] = length
            rr["table_task"] = "Full raw" if length == "full" else f"Length {length}"
            rr["table_config_key"] = exp
            rr["table_config"] = {
                "raw": "Neural Action",
                "raw_dwell": "Neural Action+Dwell",
                "raw_plus_ltl": "Neural-Symbolic Action",
                "raw_dwell_plus_ltl": "Neural-Symbolic Action+Dwell",
            }.get(exp, exp)
            rows.append(rr)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    length_rank = {L: i for i, L in enumerate(table_order)}
    out["length_rank"] = out["table_length"].map(length_rank).fillna(999)
    out["config_rank"] = out["table_config_key"].map(config_order).fillna(999)
    out = out.sort_values(["length_rank", "config_rank"]).reset_index(drop=True)
    return out


def is_best_in_group(table_df, idx, metric):
    col = f"{metric}_mean"
    if col not in table_df.columns:
        return False
    group = table_df[table_df["table_task"] == table_df.loc[idx, "table_task"]]
    vals = pd.to_numeric(group[col], errors="coerce")
    if vals.dropna().empty or pd.isna(table_df.loc[idx, col]):
        return False
    best = vals.min() if METRIC_DIRECTION.get(metric, "up") == "down" else vals.max()
    return np.isclose(float(table_df.loc[idx, col]), float(best), rtol=1e-9, atol=1e-9)


def make_latex_table(summary, metrics, out_tex, out_csv, caption, label, raw_lengths):
    table_df = prepare_table_rows(summary, raw_lengths)
    headers = ["Task", "Config", "Best Model (epoch)"]
    for m in metrics:
        arrow = r"$\uparrow$" if METRIC_DIRECTION.get(m, "up") == "up" else r"$\downarrow$"
        headers.append(f"{DISPLAY_METRIC_NAMES.get(m, m)} {arrow}")

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lll" + "c" * len(metrics) + "}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    csv_rows = []
    prev_task = None
    for idx, row in table_df.iterrows():
        if prev_task is not None and row["table_task"] != prev_task:
            lines.append(r"\midrule")
        prev_task = row["table_task"]
        model = str(row.get("model", ""))
        ep = row.get("best_epoch_mean", np.nan)
        if pd.isna(ep):
            model_cell = model
        else:
            model_cell = f"{model} ({ep:.1f})"
        cells = [latex_escape(row["table_task"]), latex_escape(row["table_config"]), latex_escape(model_cell)]
        csv_row = {"Task": row["table_task"], "Config": row["table_config"], "Best Model (epoch)": model_cell}
        for m in metrics:
            cell = fmt_mean_std(row.get(f"{m}_mean", np.nan), row.get(f"{m}_std", np.nan), m, bold=is_best_in_group(table_df, idx, m))
            cells.append(cell)
            csv_row[m] = cell.replace("$\\pm$", "±").replace(r"\textbf{", "").replace("}", "")
        lines.append(" & ".join(cells) + r" \\")
        csv_rows.append(csv_row)
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\end{table*}",
    ])
    tex = "\n".join(lines)
    Path(out_tex).write_text(tex, encoding="utf-8")
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)


def save_tables_and_summaries(runs_df, out_dir, raw_lengths):
    runs_df.to_csv(out_dir / "all_runs.csv", index=False)
    summary = make_uncertainty_summary(runs_df)
    summary = add_cost_adjusted(summary)
    summary = add_pareto_flag(summary)
    summary.to_csv(out_dir / "summary_uncertainty.csv", index=False)

    compact = summary.copy()
    if {"f1_mean", "total_pipeline_cost_sec_mean"}.issubset(compact.columns):
        compact = compact.sort_values(["f1_mean", "total_pipeline_cost_sec_mean"], ascending=[False, True])
    compact.to_csv(out_dir / "summary_cost_adjusted.csv", index=False)
    compact[compact.get("pareto_f1_cost", False) == True].to_csv(out_dir / "pareto_front.csv", index=False)

    main_metrics = ["accuracy", "balanced_accuracy", "f1", "mcc", "auc", "auprc", "brier", "ece"]
    cost_metrics = ["train_time_sec", "infer_time_sec", "model_checking_time_sec", "total_pipeline_cost_sec", "f1_per_min", "mcc_per_min", "auc_per_min", "f1_over_log_cost"]
    wide_metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "f2", "mcc", "cohen_kappa", "auc", "auprc", "brier", "logloss", "ece"]

    make_latex_table(summary, main_metrics, out_dir / "paper_table_main.tex", out_dir / "paper_table_main.csv",
                     "Predictive performance across neural, symbolic, and neural-symbolic configurations. Values are mean $\\pm$ standard deviation across seeds.",
                     "tab:main_results", raw_lengths)
    make_latex_table(summary, cost_metrics, out_dir / "paper_table_cost.tex", out_dir / "paper_table_cost.csv",
                     "Computational cost and accuracy--efficiency trade-off. Total cost includes model checking, training, and inference time.",
                     "tab:cost_results", raw_lengths)
    make_latex_table(summary, wide_metrics, out_dir / "paper_table_wide.tex", out_dir / "paper_table_wide.csv",
                     "Extended performance metrics for robustness analysis.",
                     "tab:wide_results", raw_lengths)

    return summary, compact


# ============================================================
# Main experiment
# ============================================================

def resolve_full_length(all_seed_dfs, args):
    if not args.include_full_raw:
        return None
    all_lens = []
    for df in all_seed_dfs.values():
        all_lens.extend([len(s) for s in df["_parsed_sequence"].tolist()])
    full_len = int(max(all_lens)) if all_lens else 0
    if args.full_len_cap is not None and args.full_len_cap > 0:
        full_len = min(full_len, int(args.full_len_cap))
    return full_len


def run_one_config(args, seed, config_name, experiment_type, model_name, length_label, raw_len,
                   train_df, val_df, test_df, action_vocab, z_train, z_val, z_test,
                   model_checking_time_sec, original_max_len, original_mean_len):
    y_train = train_df[args.label_col].astype(int).to_numpy()
    y_val = val_df[args.label_col].astype(int).to_numpy()
    y_test = test_df[args.label_col].astype(int).to_numpy()

    batch_size = args.batch_size
    raw_sequence_length_used = 0
    dwell_sequence_used = False

    if experiment_type == "verification_only":
        train_ds = ZDataset(z_train, y_train)
        val_ds = ZDataset(z_val, y_val)
        test_ds = ZDataset(z_test, y_test)
        model = MLPBinaryClassifier(input_dim=z_train.shape[1], hidden_dim=args.mlp_hidden_dim, dropout=args.dropout)
        mode = "z"
        raw_sequence_length_used = 0
        vocab_size = -1
        seq_len_model = z_train.shape[1]

    elif experiment_type == "raw":
        x_train = build_raw_matrix(train_df["_parsed_sequence"].tolist(), action_vocab, raw_len, args.raw_truncation)
        x_val = build_raw_matrix(val_df["_parsed_sequence"].tolist(), action_vocab, raw_len, args.raw_truncation)
        x_test = build_raw_matrix(test_df["_parsed_sequence"].tolist(), action_vocab, raw_len, args.raw_truncation)
        train_ds = RawDataset(x_train, y_train)
        val_ds = RawDataset(x_val, y_val)
        test_ds = RawDataset(x_test, y_test)
        vocab_size = len(action_vocab) + 1
        seq_len_model = raw_len
        model = TransformerSequenceClassifier(vocab_size=vocab_size, max_len=raw_len, d_model=args.d_model,
                                              n_heads=args.n_heads, n_layers=args.n_layers,
                                              dim_feedforward=args.dim_feedforward, dropout=args.dropout)
        mode = "raw"
        raw_sequence_length_used = raw_len

    elif experiment_type == "raw_dwell":
        x_train = build_raw_matrix(train_df["_parsed_sequence"].tolist(), action_vocab, raw_len, args.raw_truncation)
        x_val = build_raw_matrix(val_df["_parsed_sequence"].tolist(), action_vocab, raw_len, args.raw_truncation)
        x_test = build_raw_matrix(test_df["_parsed_sequence"].tolist(), action_vocab, raw_len, args.raw_truncation)
        mean, std = fit_dwell_scaler(train_df["_dwell_seconds_sequence"].tolist(), raw_len, args.raw_truncation, log1p=args.log1p_dwell)
        d_train = build_dwell_matrix(train_df["_dwell_seconds_sequence"].tolist(), raw_len, args.raw_truncation, mean, std, log1p=args.log1p_dwell)
        d_val = build_dwell_matrix(val_df["_dwell_seconds_sequence"].tolist(), raw_len, args.raw_truncation, mean, std, log1p=args.log1p_dwell)
        d_test = build_dwell_matrix(test_df["_dwell_seconds_sequence"].tolist(), raw_len, args.raw_truncation, mean, std, log1p=args.log1p_dwell)
        train_ds = RawDwellDataset(x_train, d_train, y_train)
        val_ds = RawDwellDataset(x_val, d_val, y_val)
        test_ds = RawDwellDataset(x_test, d_test, y_test)
        vocab_size = len(action_vocab) + 1
        seq_len_model = raw_len
        model = TransformerDwellSequenceClassifier(vocab_size=vocab_size, max_len=raw_len, d_model=args.d_model,
                                                   n_heads=args.n_heads, n_layers=args.n_layers,
                                                   dim_feedforward=args.dim_feedforward, dropout=args.dropout)
        mode = "raw_dwell"
        raw_sequence_length_used = raw_len
        dwell_sequence_used = True

    elif experiment_type == "raw_plus_ltl":
        x_train = build_raw_matrix(train_df["_parsed_sequence"].tolist(), action_vocab, raw_len, args.raw_truncation)
        x_val = build_raw_matrix(val_df["_parsed_sequence"].tolist(), action_vocab, raw_len, args.raw_truncation)
        x_test = build_raw_matrix(test_df["_parsed_sequence"].tolist(), action_vocab, raw_len, args.raw_truncation)
        train_ds = FusionDataset(x_train, z_train, y_train)
        val_ds = FusionDataset(x_val, z_val, y_val)
        test_ds = FusionDataset(x_test, z_test, y_test)
        vocab_size = len(action_vocab) + 1
        seq_len_model = raw_len
        model = LateFusionTransformerLTLClassifier(vocab_size=vocab_size, max_len=raw_len, z_dim=z_train.shape[1],
                                                   d_model=args.d_model, n_heads=args.n_heads,
                                                   n_layers=args.n_layers, dim_feedforward=args.dim_feedforward,
                                                   ltl_hidden_dim=args.late_ltl_hidden_dim,
                                                   fusion_hidden_dim=args.late_fusion_hidden_dim,
                                                   dropout=args.dropout)
        mode = "fusion"
        raw_sequence_length_used = raw_len

    elif experiment_type == "raw_dwell_plus_ltl":
        x_train = build_raw_matrix(train_df["_parsed_sequence"].tolist(), action_vocab, raw_len, args.raw_truncation)
        x_val = build_raw_matrix(val_df["_parsed_sequence"].tolist(), action_vocab, raw_len, args.raw_truncation)
        x_test = build_raw_matrix(test_df["_parsed_sequence"].tolist(), action_vocab, raw_len, args.raw_truncation)
        mean, std = fit_dwell_scaler(train_df["_dwell_seconds_sequence"].tolist(), raw_len, args.raw_truncation, log1p=args.log1p_dwell)
        d_train = build_dwell_matrix(train_df["_dwell_seconds_sequence"].tolist(), raw_len, args.raw_truncation, mean, std, log1p=args.log1p_dwell)
        d_val = build_dwell_matrix(val_df["_dwell_seconds_sequence"].tolist(), raw_len, args.raw_truncation, mean, std, log1p=args.log1p_dwell)
        d_test = build_dwell_matrix(test_df["_dwell_seconds_sequence"].tolist(), raw_len, args.raw_truncation, mean, std, log1p=args.log1p_dwell)
        train_ds = FusionDwellDataset(x_train, d_train, z_train, y_train)
        val_ds = FusionDwellDataset(x_val, d_val, z_val, y_val)
        test_ds = FusionDwellDataset(x_test, d_test, z_test, y_test)
        vocab_size = len(action_vocab) + 1
        seq_len_model = raw_len
        model = LateFusionDwellTransformerLTLClassifier(vocab_size=vocab_size, max_len=raw_len, z_dim=z_train.shape[1],
                                                        d_model=args.d_model, n_heads=args.n_heads,
                                                        n_layers=args.n_layers, dim_feedforward=args.dim_feedforward,
                                                        ltl_hidden_dim=args.late_ltl_hidden_dim,
                                                        fusion_hidden_dim=args.late_fusion_hidden_dim,
                                                        dropout=args.dropout)
        mode = "fusion_dwell"
        raw_sequence_length_used = raw_len
        dwell_sequence_used = True

    else:
        raise ValueError(f"Unknown experiment_type: {experiment_type}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=args.pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=args.pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=args.pin_memory)

    set_seed(seed)
    metrics = train_torch_model(model, train_loader, val_loader, test_loader, y_train, args, mode=mode)

    mc_time = 0.0 if experiment_type in {"raw", "raw_dwell"} else float(model_checking_time_sec)
    total_train_infer = metrics["train_time_sec"] + metrics["infer_time_sec"]
    total_feature_train = mc_time + metrics["train_time_sec"]
    total_pipeline = mc_time + metrics["train_time_sec"] + metrics["infer_time_sec"]
    eps = 1e-8

    feature_pack = {
        "raw": "raw_behavior_only",
        "raw_dwell": "raw_behavior_plus_dwell_seconds",
        "verification_only": "auto_ltl_verification_matrix",
        "raw_plus_ltl": "raw_behavior_plus_ltl",
        "raw_dwell_plus_ltl": "raw_behavior_dwell_plus_ltl",
    }.get(experiment_type, experiment_type)

    metrics.update({
        "run_id": f"{experiment_type}__{model_name}__len_{length_label}__seed_{seed}",
        "seed": seed,
        "feature_pack": feature_pack,
        "experiment_type": experiment_type,
        "config_name": config_name,
        "model": model_name,
        "length_label": str(length_label),
        "raw_len": int(raw_len) if raw_len is not None else -1,
        "raw_sequence_length_used": int(raw_sequence_length_used),
        "dwell_sequence_used": bool(dwell_sequence_used),
        "dwell_seconds_col": args.dwell_seconds_col if dwell_sequence_used else "",
        "original_max_sequence_length": int(original_max_len),
        "original_mean_sequence_length": float(original_mean_len),
        "seq_len_model": int(seq_len_model),
        "vocab_size": int(vocab_size),
        "model_checking_time_sec": float(mc_time),
        "shared_seed_model_checking_time_sec": float(model_checking_time_sec),
        "total_train_infer_time_sec": float(total_train_infer),
        "total_feature_time_plus_train_sec": float(total_feature_train),
        "total_pipeline_cost_sec": float(total_pipeline),
        "f1_per_min": metrics["f1"] / (total_pipeline / 60.0 + eps),
        "mcc_per_min": metrics["mcc"] / (total_pipeline / 60.0 + eps),
        "auc_per_min": metrics["auc"] / (total_pipeline / 60.0 + eps),
        "f1_over_log_cost": metrics["f1"] / (np.log1p(total_pipeline) + eps),
        "mcc_over_log_cost": metrics["mcc"] / (np.log1p(total_pipeline) + eps),
        "auc_over_log_cost": metrics["auc"] / (np.log1p(total_pipeline) + eps),
        "mc_window": int(args.mc_window),
        "mc_truncation": args.mc_truncation,
        "mc_tag": model_check_tag(args.mc_window, args.mc_truncation),
        "ltl_representation": getattr(args, "ltl_representation", "binary"),
        "ltl_representation_tag": ltl_representation_tag(args),
        "status": "ok",
        "error": "",
    })
    return metrics

def run_experiment(args):
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or f"integrated_pytorch_{ltl_representation_tag(args)}_top{args.top_k}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Output directory:", out_dir)
    print("MC setting:", model_check_tag(args.mc_window, args.mc_truncation), "mc_window=", args.mc_window, "mc_truncation=", args.mc_truncation)
    print("LTL representation:", args.ltl_representation)

    with open(out_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        cfg = vars(args).copy()
        cfg["action_token_map"] = DEFAULT_ACTION_TOKEN_MAP
        cfg["mc_tag"] = model_check_tag(args.mc_window, args.mc_truncation)
        cfg["ltl_representation_tag"] = ltl_representation_tag(args)
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # Preload all seed data to determine global full length.
    seed_dfs = {}
    length_rows = []
    for seed in args.seeds:
        df, rows = load_seed_split_data(args, seed)
        seed_dfs[seed] = df
        length_rows.extend(rows)
    pd.DataFrame(length_rows).to_csv(out_dir / "sequence_length_stats.csv", index=False)

    full_len = resolve_full_length(seed_dfs, args)
    if full_len is not None:
        print("Global full raw baseline length:", full_len)

    all_rows = []
    mc_rows = []
    all_rule_stats = []

    for seed in args.seeds:
        print("\n" + "=" * 90)
        print(f"Seed {seed}")
        print("=" * 90)

        seed_df = seed_dfs[seed]
        train_df = seed_df[seed_df["_split"] == "train"].reset_index(drop=True)
        val_df = seed_df[seed_df["_split"] == "val"].reset_index(drop=True)
        test_df = seed_df[seed_df["_split"] == "test"].reset_index(drop=True)

        all_sequences = seed_df["_logic_trace"].tolist()
        labels_all = seed_df[args.label_col].astype(int).to_numpy()
        session_ids_all = seed_df[args.session_col].astype(str).to_numpy()
        split_labels_all = seed_df["_split"].astype(str).to_numpy()
        original_lengths = np.array([len(s) for s in all_sequences])
        original_max_len = int(original_lengths.max())
        original_mean_len = float(original_lengths.mean())

        action_vocab = build_action_vocab(train_df["_parsed_sequence"].tolist())
        print("Action vocab:", action_vocab)
        print(f"Train/Val/Test sizes: {len(train_df)}, {len(val_df)}, {len(test_df)}")
        print(f"Positive rate train/val/test: {train_df[args.label_col].mean():.4f}, {val_df[args.label_col].mean():.4f}, {test_df[args.label_col].mean():.4f}")
        print(f"Original length mean={original_mean_len:.2f}, p95={np.percentile(original_lengths, 95):.1f}, max={original_max_len}")

        formula_json_path = formula_path_for_seed(args, seed)
        if not formula_json_path.exists():
            raise FileNotFoundError(f"Missing formula JSON for seed {seed}: {formula_json_path}")
        print("Formula JSON:", formula_json_path)
        formula_items, trees = load_formula_records(formula_json_path, top_k=args.top_k)

        seed_out_dir = out_dir / f"seed_{seed}"
        file_stem = f"Z_seed{seed}_{ltl_representation_tag(args)}_top{args.top_k}_semantic_unique"
        z_npz_path = seed_out_dir / f"{file_stem}.npz"

        if args.reuse_z and z_npz_path.exists():
            print("Reusing Z:", z_npz_path)
            pack = np.load(z_npz_path, allow_pickle=True)
            z_all = pack["Z"].astype(np.float32)
            model_checking_time_sec = float(pack["model_checking_time_sec"][0]) if "model_checking_time_sec" in pack.files else 0.0
            if "feature_names" in pack.files:
                feature_names = [str(x) for x in pack["feature_names"].tolist()]
            else:
                feature_names = [f"F{k:03d}" for k in range(z_all.shape[1])]
            representation_meta = {"representation": getattr(args, "ltl_representation", "binary"), "reused": True}
        else:
            z_all, model_checking_time_sec, feature_names, representation_meta = compute_ltl_representation_from_trees(
                sequences=all_sequences,
                trees=trees,
                args=args,
            )
            save_z_artifacts(seed_out_dir, file_stem, z_all, session_ids_all, labels_all, split_labels_all,
                             formula_items, model_checking_time_sec, args, formula_json_path,
                             feature_names=feature_names, representation_meta=representation_meta)

        rule_stats_path = seed_out_dir / f"{file_stem}_rule_stats.csv"
        if rule_stats_path.exists():
            rs = pd.read_csv(rule_stats_path)
            rs.insert(0, "seed", seed)
            rs.insert(1, "formula_json", str(formula_json_path))
            all_rule_stats.append(rs)

        mc_rows.append({
            "seed": seed,
            "formula_mode": args.formula_mode,
            "formula_json": str(formula_json_path),
            "n_formulas": int(z_all.shape[1]),
            "Z_rows": int(z_all.shape[0]),
            "Z_cols": int(z_all.shape[1]),
            "Z_mean": float(z_all.mean()),
            "model_checking_time_sec": float(model_checking_time_sec),
            "mc_window": int(args.mc_window),
            "mc_tag": model_check_tag(args.mc_window, args.mc_truncation),
            "ltl_representation": args.ltl_representation,
            "ltl_representation_tag": ltl_representation_tag(args),
            "mc_truncation": args.mc_truncation,
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "npz_path": str(z_npz_path),
        })
        pd.DataFrame(mc_rows).to_csv(out_dir / "model_checking_by_seed.csv", index=False)

        # Split Z by row order from seed_df.
        z_train = z_all[seed_df["_split"].values == "train"]
        z_val = z_all[seed_df["_split"].values == "val"]
        z_test = z_all[seed_df["_split"].values == "test"]

        planned = []
        if args.run_verification_only:
            planned.append(("Symbolic", "verification_only", "mlp", "verification_only", None))

        # Main raw baseline for this stage: action + dwell_seconds.
        if args.run_raw:
            for L in args.raw_lengths:
                planned.append(("Neural Action+Dwell", "raw_dwell", "transformer_dwell", str(L), int(L)))
            if args.include_full_raw and full_len is not None:
                planned.append(("Neural Action+Dwell", "raw_dwell", "transformer_dwell", "full", int(full_len)))

        # Optional action-only raw baseline for debugging/ablation. Off by default.
        if args.run_raw_action_only:
            for L in args.raw_lengths:
                planned.append(("Neural Action", "raw", "transformer", str(L), int(L)))
            if args.include_full_raw and full_len is not None:
                planned.append(("Neural Action", "raw", "transformer", "full", int(full_len)))

        if args.run_raw_plus_ltl:
            for L in args.raw_lengths:
                # Behavior-only + verification.
                planned.append(("Neural-Symbolic Action", "raw_plus_ltl", "latefusion_transformer_mlp", str(L), int(L)))
                # Behavior + dwell_seconds + verification.
                if args.run_raw_dwell_plus_ltl:
                    planned.append(("Neural-Symbolic Action+Dwell", "raw_dwell_plus_ltl", "latefusion_dwell_transformer_mlp", str(L), int(L)))

        for config_name, experiment_type, model_name, length_label, raw_len in planned:
            print(f"\nRunning seed={seed} {experiment_type} length={length_label} model={model_name}")
            try:
                row = run_one_config(args, seed, config_name, experiment_type, model_name, length_label, raw_len,
                                     train_df, val_df, test_df, action_vocab, z_train, z_val, z_test,
                                     model_checking_time_sec, original_max_len, original_mean_len)
                all_rows.append(row)
                print(f"  f1={row['f1']:.4f} mcc={row['mcc']:.4f} auc={row['auc']:.4f} brier={row['brier']:.4f} total_cost={row['total_pipeline_cost_sec']:.2f}s")
            except Exception as e:
                print("  FAILED:", repr(e))
                all_rows.append({
                    "run_id": f"{experiment_type}__{model_name}__len_{length_label}__seed_{seed}",
                    "seed": seed,
                    "feature_pack": "auto_ltl_verification_matrix" if experiment_type != "raw" else "raw_behavior_only",
                    "experiment_type": experiment_type,
                    "config_name": config_name,
                    "model": model_name,
                    "length_label": str(length_label),
                    "mc_tag": model_check_tag(args.mc_window, args.mc_truncation),
                    "ltl_representation": getattr(args, "ltl_representation", "binary"),
                    "ltl_representation_tag": ltl_representation_tag(args),
                    "status": "failed",
                    "error": repr(e),
                })
            pd.DataFrame(all_rows).to_csv(out_dir / "all_runs.csv", index=False)

    if all_rule_stats:
        pd.concat(all_rule_stats, ignore_index=True).to_csv(out_dir / "all_rule_stats.csv", index=False)

    runs_df = pd.DataFrame(all_rows)
    summary, compact = save_tables_and_summaries(runs_df, out_dir, args.raw_lengths)

    print("\nDone.")
    print("Output directory:", out_dir)
    print("Top summary rows:")
    with pd.option_context("display.max_columns", 30):
        print(compact.head(20).to_string(index=False))


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--aligned_splits_dir", type=str, default="/content/aligned_splits_v1")
    p.add_argument("--out_dir", type=str, default="/content/drive/MyDrive/data/ltl_integrated_pytorch_results")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45])

    p.add_argument("--seq_col", type=str, default="behavior_sequence", help="Action sequence column used by the neural raw encoder.")
    p.add_argument("--dwell_seconds_col", type=str, default="dwell_seconds_sequence", help="Numeric dwell-time sequence column used by raw+dwell baselines.")
    p.add_argument("--dwell_bucket_col", type=str, default="dwell_bucket_sequence", help="Dwell-bucket sequence column used to build multi-proposition traces if logic_trace is unavailable.")
    p.add_argument("--logic_trace_col", type=str, default="logic_trace", help="Optional TLTLf trace column, e.g. [[b,dm],[d,dh]]. Used for model checking.")
    p.add_argument("--require_dwell_seconds", action="store_true", default=True)
    p.add_argument("--allow_missing_dwell_seconds", action="store_false", dest="require_dwell_seconds")
    p.add_argument("--log1p_dwell", action="store_true", default=True)
    p.add_argument("--no_log1p_dwell", action="store_false", dest="log1p_dwell")
    p.add_argument("--label_col", type=str, default="purchase")
    p.add_argument("--session_col", type=str, default="session_id_hash")

    p.add_argument("--formula_mode", type=str, default="fixed", choices=["fixed", "seed_specific"])
    p.add_argument("--formula_source_seed", type=int, default=42)
    p.add_argument("--formula_json", type=str, default=None)
    p.add_argument("--formula_json_pattern", type=str, default="{aligned_splits_dir}/seed_{seed}/window_last{window}/formulas/top20_semantic_unique_formulas.json")
    p.add_argument("--formula_window", type=int, default=30)
    p.add_argument("--top_k", type=int, default=20)

    # Default MC is full trace.
    p.add_argument("--mc_window", type=int, default=0, help="0 or negative = full trace. Positive N = N-event window.")
    p.add_argument("--mc_truncation", type=str, default="last", choices=["first", "last"], help="Used only when mc_window > 0. first = prefix, last = suffix.")
    p.add_argument("--remove_purchase_tokens", action="store_true", default=True)
    p.add_argument("--keep_purchase_tokens", action="store_false", dest="remove_purchase_tokens")
    p.add_argument("--reuse_z", action="store_true")
    p.add_argument("--progress_every", type=int, default=50000)
    p.add_argument("--ltl_representation", type=str, default="binary", choices=["binary", "automaton", "automaton_no_binary"],
                   help="binary = old final satisfaction; automaton = residual automaton-state/progression features plus binary; automaton_no_binary = residual-state features only.")
    p.add_argument("--max_states_per_formula", type=int, default=64,
                   help="Maximum residual automaton states to one-hot per formula. Extra states are collapsed to OTHER.")

    p.add_argument("--raw_lengths", nargs="+", type=int, default=[10, 20, 30])
    p.add_argument("--raw_truncation", type=str, default="last", choices=["first", "last"])
    p.add_argument("--include_full_raw", action="store_true", default=True)
    p.add_argument("--no_full_raw", action="store_false", dest="include_full_raw")
    p.add_argument("--full_len_cap", type=int, default=None)

    p.add_argument("--run_raw", action="store_true", default=True, help="Run the main raw baseline: behavior_sequence + dwell_seconds_sequence.")
    p.add_argument("--no_raw", action="store_false", dest="run_raw")
    p.add_argument("--run_raw_action_only", action="store_true", default=False, help="Optional: also run behavior_sequence-only raw baselines.")
    p.add_argument("--run_verification_only", action="store_true", default=True)
    p.add_argument("--no_verification_only", action="store_false", dest="run_verification_only")
    p.add_argument("--run_raw_plus_ltl", action="store_true", default=True, help="Run behavior_sequence-only + verification.")
    p.add_argument("--no_raw_plus_ltl", action="store_false", dest="run_raw_plus_ltl")
    p.add_argument("--run_raw_dwell_plus_ltl", action="store_true", default=True, help="Run behavior_sequence + dwell_seconds_sequence + verification.")
    p.add_argument("--no_raw_dwell_plus_ltl", action="store_false", dest="run_raw_dwell_plus_ltl")

    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--threshold_metric", type=str, default="f1", choices=["f1", "mcc"])
    p.add_argument("--early_stop_metric", type=str, default="f1")

    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--dim_feedforward", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--mlp_hidden_dim", type=int, default=64)
    p.add_argument("--late_ltl_hidden_dim", type=int, default=64)
    p.add_argument("--late_fusion_hidden_dim", type=int, default=128)

    p.add_argument("--cpu", action="store_true")
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
