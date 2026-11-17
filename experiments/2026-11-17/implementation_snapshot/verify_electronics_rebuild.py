"""Verify the locally rebuilt Electronics data against its Colab evidence."""
import argparse
import ast
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from diagnose_coveo import sha, vector


ROOT = Path(__file__).resolve().parent
DEFAULT_REBUILD = ROOT.parents[1] / "work" / "electronics_rebuild"
DEFAULT_BASIS = Path(
    "C:/Users/ycw/OneDrive - Høgskulen på Vestlandet/Desktop/LTL experiment/"
    "dataset2/BP_semantic_unique_top10_formulas.json"
)
DWELL_ATOMS = {"dl", "dm", "dh", "d_short", "d_medium", "d_long"}


def load_split(ready, seed, split):
    return pd.read_csv(
        ready / "electronics_splits" / f"seed_{seed}" / f"{split}.csv",
        dtype={"session_id_hash": str, "user_id": str},
    )


def compare_notebook_dwell_summaries(notebook_log, dwell_fix):
    """Compare split-specific numeric evidence even when old ID files are missing."""
    block = notebook_log.read_text(encoding="utf-8").split("CELL 20 SOURCE", 1)[1]
    block = block.split("CELL 21 SOURCE", 1)[0]
    pattern = r"^ - .*?/electronics_ready_noleak/([^\r\n]+)\r?\n\s+before:\s+(\{[^\r\n]+\})"
    old = {path: ast.literal_eval(stats) for path, stats in re.findall(pattern, block, re.M)}
    rows = []
    for entry in dwell_fix["patched_csvs"]:
        path = str(entry["csv"]).replace("\\", "/").split("/electronics_ready_noleak/", 1)[-1]
        if path not in old:
            continue
        old_stats = old[path]
        new_stats = entry["before"]
        matches = all(np.isclose(old_stats[key], new_stats[key], atol=1e-10, rtol=0)
                      for key in old_stats)
        rows.append({
            "file": path, "dwell_summary_matches": matches,
            "colab_positive_last_dwell_mean": old_stats["label_1_last_dwell_mean"],
            "rebuilt_positive_last_dwell_mean": new_stats["label_1_last_dwell_mean"],
        })
    if len(rows) != 13:
        raise ValueError("Expected Colab dwell evidence for master CSV plus 12 split CSVs")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", type=Path, default=DEFAULT_REBUILD)
    parser.add_argument("--basis", type=Path, default=DEFAULT_BASIS)
    parser.add_argument("--notebook-log", type=Path,
                        default=DEFAULT_REBUILD.parent / "electronics_nb_1_cells.txt")
    parser.add_argument("--out", type=Path, default=ROOT / "electronics_rebuild")
    args = parser.parse_args()
    ready = args.rebuild / "electronics_ready_noleak"
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.rebuild / "source" / "events.csv"
    sessions_path = ready / "electronics_sessions_ltl_ready.csv"
    dwell_fix_path = ready / "dwell_leakage_fix_report.json"
    dwell_fix = json.loads(dwell_fix_path.read_text(encoding="utf-8"))
    if (dwell_fix["new_dwell_q1_sec"], dwell_fix["new_dwell_q2_sec"]) != (39.0, 171.0):
        raise ValueError("Rebuilt dwell thresholds differ from the Colab log")
    dwell_evidence = compare_notebook_dwell_summaries(args.notebook_log, dwell_fix)
    pd.DataFrame(dwell_evidence).to_csv(args.out / "notebook_dwell_comparison.csv", index=False)
    sessions = pd.read_csv(sessions_path, dtype={"session_id_hash": str, "user_id": str})
    if (len(sessions), int(sessions.purchase.sum())) != (138995, 21062):
        raise ValueError("Rebuilt session/positive totals differ from the Colab log")
    if sessions.session_id_hash.duplicated().any():
        raise ValueError("Duplicate master session IDs")
    if any(json.loads(value)[-1] != 0 for value in sessions.dwell_time_sequence):
        raise ValueError("Rebuilt sessions contain nonzero terminal dwell")
    for value in sessions.action_sequence:
        if {"p", "purchase"} & set(json.loads(value)):
            raise ValueError("Unexpected purchase event in the input")

    # The final notebook samples each class with seed 42 and selects on this same sample.
    train42 = load_split(ready, 42, "train")
    sample = {
        label: train42[train42.purchase == label].sample(n=4000, random_state=42)
        for label in (1, 0)
    }
    pool_ids = set(pd.concat(sample.values()).session_id_hash)
    pool_users = set(pd.concat(sample.values()).user_id)
    basis = json.loads(args.basis.read_text(encoding="utf-8"))
    trees = [ast.literal_eval(record["formula_tree"]) for record in basis]
    traces = {
        label: [
            [set(state) - DWELL_ATOMS for state in json.loads(value)[-10:]]
            for value in frame.atom_sequence
        ]
        for label, frame in sample.items()
    }
    support = {
        label: np.stack([vector(trace, trees) for trace in group]).mean(axis=0)
        for label, group in traces.items()
    }
    support_rows = []
    for j, record in enumerate(basis):
        support_rows.append({
            "formula_id": record["formula_id"],
            "formula": record["formula_str"],
            "saved_positive_support": record["pos_support"],
            "rebuilt_positive_support": support[1][j],
            "saved_negative_support": record["neg_support"],
            "rebuilt_negative_support": support[0][j],
            "support_matches": bool(
                np.isclose(support[1][j], record["pos_support"], atol=1e-12, rtol=0)
                and np.isclose(support[0][j], record["neg_support"], atol=1e-12, rtol=0)
            ),
        })
    pd.DataFrame(support_rows).to_csv(args.out / "bp_support_replay.csv", index=False)
    all_atoms = sorted(set().union(*(state for group in traces.values() for trace in group for state in trace)))
    pool_path = args.rebuild / "formula_learning_noleak_4000_last10_seed42" / "BP"
    pool_path.mkdir(parents=True, exist_ok=True)
    pd.concat(sample.values())[["session_id_hash", "user_id", "purchase"]].to_csv(
        pool_path / "formula_pool_ids.csv", index=False
    )
    training_json = {
        "vocab": all_atoms,
        "traces_pos": [[sorted(state) for state in trace] for trace in traces[1]],
        "traces_neg": [[sorted(state) for state in trace] for trace in traces[0]],
        "ltlftree": "",
    }
    (pool_path / "BP_train_seed42_pos4000_neg4000_last10.json").write_text(
        json.dumps(training_json), encoding="utf-8"
    )

    split_rows = []
    user_rows = []
    split_hashes = {}
    expected = {"train": (73717, 14743), "val": (10531, 2106), "test": (21062, 4213)}
    for seed in (42, 43, 44, 45):
        ids = {}
        users = {}
        for split in expected:
            frame = load_split(ready, seed, split)
            if (len(frame), int(frame.purchase.sum())) != expected[split]:
                raise ValueError(f"Rebuilt seed {seed} {split} counts differ from Colab")
            ids[split] = set(frame.session_id_hash)
            users[split] = set(frame.user_id)
            split_hashes[f"{seed}_{split}"] = sha(
                ready / "electronics_splits" / f"seed_{seed}" / f"{split}.csv"
            )
            split_rows.append({
                "seed": seed, "split": split, "n": len(frame),
                "positive": int(frame.purchase.sum()), "users": len(users[split]),
                "formula_pool_session_overlap": len(ids[split] & pool_ids),
                "formula_pool_user_overlap": len(users[split] & pool_users),
                "sessions_longer_than_512": int((frame.session_len > 512).sum()),
                "sessions_longer_than_10": int((frame.session_len > 10).sum()),
            })
        for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
            if ids[first] & ids[second]:
                raise ValueError(f"Session overlap in seed {seed}: {first}/{second}")
            user_rows.append({
                "seed": seed, "first_split": first, "second_split": second,
                "user_overlap": len(users[first] & users[second]),
            })
    pd.DataFrame(split_rows).to_csv(args.out / "split_audit.csv", index=False)
    pd.DataFrame(user_rows).to_csv(args.out / "user_overlap.csv", index=False)
    manifest = {
        "source_url": "https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-electronics-store",
        "source_csv_sha256": sha(source),
        "ready_dir": str(ready.resolve()),
        "sessions_sha256": sha(sessions_path),
        "basis_sha256": sha(args.basis),
        "split_hashes": split_hashes,
        "session_count": len(sessions),
        "positive_count": int(sessions.purchase.sum()),
        "session_length_max": int(sessions.session_len.max()),
        "sessions_longer_than_512": int((sessions.session_len > 512).sum()),
        "dwell_fix_report_sha256": sha(dwell_fix_path),
        "dwell_threshold_seconds": [dwell_fix["new_dwell_q1_sec"], dwell_fix["new_dwell_q2_sec"]],
        "colab_dwell_summaries_matched": sum(row["dwell_summary_matches"] for row in dwell_evidence),
        "colab_dwell_summaries_total": len(dwell_evidence),
        "formula_pool_sessions": len(pool_ids),
        "formula_pool_vocabulary": all_atoms,
        "formula_support_pairs_matched": sum(row["support_matches"] for row in support_rows),
        "formula_support_pairs_total": len(basis),
        "historical_exact_id_match": "unverified; original split CSVs unavailable",
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
