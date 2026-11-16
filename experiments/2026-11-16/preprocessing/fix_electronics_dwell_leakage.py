#!/usr/bin/env python3
"""
fix_electronics_dwell_leakage.py

Hotfix for the Kaggle electronics session files produced by electronics_ltlf_pipeline.py.

Problem fixed:
  In the earlier prepared data, dwell was computed before purchase truncation.
  Therefore, for positive sessions, the last pre-purchase event could contain
  the time gap to the hidden purchase event, while negative sessions ended with
  dwell=0. This leaks purchase information into B+D / ADPC raw models.

What this script does:
  1. Copies an existing electronics_ready directory to a new out_dir.
  2. For electronics_sessions_ltl_ready.csv and every split CSV, sets the last
     dwell value of each input trace to 0.0.
  3. Recomputes dwell buckets from the patched master CSV.
  4. Updates dwell_bucket_sequence and replaces dl/dm/dh atoms inside atom_sequence.
  5. Removes stale model-checking matrices/metadata from the output directory.

After running this, use --sessions_csv and --splits_dir from the fixed out_dir.
For any LTL formulas that include dwell atoms, rerun modelcheck on the fixed CSV.
"""

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd

DWELL_ATOMS = {"dl", "dm", "dh"}


def loads_list(x: Any):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    return json.loads(x)


def dumps_list(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False)


def set_last_dwell_zero(seq):
    vals = [float(v) for v in loads_list(seq)]
    if vals:
        vals[-1] = 0.0
    return vals


def bucket_dwell(vals: List[float], q1: float, q2: float) -> List[str]:
    out = []
    for x in vals:
        x = float(x)
        if x <= q1:
            out.append("dl")
        elif x <= q2:
            out.append("dm")
        else:
            out.append("dh")
    return out


def update_atom_sequence(atom_seq, dwell_buckets):
    steps = loads_list(atom_seq)
    new_steps = []
    for i, atoms in enumerate(steps):
        if not isinstance(atoms, list):
            atoms = list(atoms)
        s = [str(a) for a in atoms if str(a) not in DWELL_ATOMS]
        if i < len(dwell_buckets):
            s.append(str(dwell_buckets[i]))
        new_steps.append(sorted(set(s)))
    return new_steps


def last_dwell_diagnostic(df: pd.DataFrame, label_col: str = "purchase") -> dict:
    last_vals = []
    for seq in df["dwell_time_sequence"]:
        vals = [float(v) for v in loads_list(seq)]
        last_vals.append(vals[-1] if vals else np.nan)
    tmp = pd.DataFrame({"last_dwell": last_vals, "purchase": df[label_col].astype(int)})
    out = {}
    for y in [0, 1]:
        sub = tmp[tmp["purchase"] == y]["last_dwell"].dropna()
        if len(sub) == 0:
            continue
        out[f"label_{y}_n"] = int(len(sub))
        out[f"label_{y}_last_dwell_mean"] = float(sub.mean())
        out[f"label_{y}_last_dwell_median"] = float(sub.median())
        out[f"label_{y}_last_dwell_gt0_rate"] = float((sub > 0).mean())
    return out


def compute_thresholds_from_master(master_csv: Path) -> tuple[float, float]:
    df = pd.read_csv(master_csv)
    positive_dwell = []
    for seq in df["dwell_time_sequence"]:
        vals = set_last_dwell_zero(seq)
        positive_dwell.extend([v for v in vals if v > 0])
    if len(positive_dwell) == 0:
        return 1.0, 5.0
    q1, q2 = np.quantile(positive_dwell, [1/3, 2/3]).tolist()
    return float(q1), float(q2)


def patch_csv(csv_path: Path, q1: float, q2: float) -> dict:
    df = pd.read_csv(csv_path)
    before = last_dwell_diagnostic(df) if "dwell_time_sequence" in df.columns and "purchase" in df.columns else {}

    if "dwell_time_sequence" not in df.columns:
        return {"csv": str(csv_path), "patched": False, "reason": "no dwell_time_sequence"}

    new_dwell = []
    new_buckets = []
    new_atoms = []

    has_atoms = "atom_sequence" in df.columns
    for _, row in df.iterrows():
        vals = set_last_dwell_zero(row["dwell_time_sequence"])
        buckets = bucket_dwell(vals, q1, q2)
        new_dwell.append(dumps_list(vals))
        new_buckets.append(dumps_list(buckets))
        if has_atoms:
            new_atoms.append(dumps_list(update_atom_sequence(row["atom_sequence"], buckets)))

    df["dwell_time_sequence"] = new_dwell
    df["dwell_bucket_sequence"] = new_buckets
    if has_atoms:
        df["atom_sequence"] = new_atoms

    df.to_csv(csv_path, index=False)
    after = last_dwell_diagnostic(df) if "purchase" in df.columns else {}

    return {"csv": str(csv_path), "patched": True, "before": before, "after": after}


def remove_stale_outputs(out_dir: Path):
    patterns = [
        "Z_*_electronics.npz",
        "Z_*_electronics.meta.json",
        "Z_*_electronics.formula_support.csv",
        "*.formula_support.csv",
    ]
    removed = []
    for pat in patterns:
        for p in out_dir.glob(pat):
            p.unlink(missing_ok=True)
            removed.append(str(p))
    ltlf_dir = out_dir / "ltlf_matrices"
    if ltlf_dir.exists():
        shutil.rmtree(ltlf_dir)
        removed.append(str(ltlf_dir))
    return removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ready_dir", required=True, help="Existing electronics_ready directory, possibly leaky.")
    ap.add_argument("--out_dir", required=True, help="Output fixed ready directory.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite out_dir if it exists.")
    args = ap.parse_args()

    ready_dir = Path(args.ready_dir)
    out_dir = Path(args.out_dir)
    if not ready_dir.exists():
        raise FileNotFoundError(f"ready_dir not found: {ready_dir}")
    if out_dir.exists():
        if args.overwrite:
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(f"out_dir already exists: {out_dir}. Use --overwrite if intended.")

    shutil.copytree(ready_dir, out_dir)

    master_csv = out_dir / "electronics_sessions_ltl_ready.csv"
    if not master_csv.exists():
        raise FileNotFoundError(f"Cannot find master CSV: {master_csv}")

    q1, q2 = compute_thresholds_from_master(master_csv)
    print(f"New dwell bucket thresholds after last-step zeroing: q1={q1:.6f}, q2={q2:.6f}")

    csvs = [master_csv]
    split_root = out_dir / "electronics_splits"
    if split_root.exists():
        csvs.extend(sorted(split_root.glob("seed_*/*.csv")))

    reports = []
    for p in csvs:
        reports.append(patch_csv(p, q1, q2))

    removed = remove_stale_outputs(out_dir)

    meta = {
        "source_ready_dir": str(ready_dir),
        "fixed_ready_dir": str(out_dir),
        "fix": "set final input-step dwell to 0 and recompute dwell buckets/atoms",
        "new_dwell_q1_sec": q1,
        "new_dwell_q2_sec": q2,
        "patched_csvs": reports,
        "removed_stale_modelcheck_outputs": removed,
    }
    with open(out_dir / "dwell_leakage_fix_report.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Patched CSVs:")
    for r in reports:
        print(" -", r["csv"])
        if "before" in r:
            print("   before:", r["before"])
            print("   after :", r["after"])
    print("Removed stale model-checking outputs:", len(removed))
    print("Saved report:", out_dir / "dwell_leakage_fix_report.json")


if __name__ == "__main__":
    main()
