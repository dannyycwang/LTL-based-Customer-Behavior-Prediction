"""Audit the saved Cosmetics splits and BP formula sample against Drive inputs."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from diagnose_coveo import sha, vector
from fix_cosmetics_parser import parse_event_trace


ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1] / "work"
OUT = ROOT / "cosmetics_seed42_audit"
SPLIT_NAMES = ("train", "val", "test")


def read_sessions(path):
    return pd.read_csv(
        path,
        usecols=["session_id_hash", "user_id", "purchase", "prop_BDP", "n_events", "behavior_sequence"],
        dtype={"session_id_hash": str, "user_id": str},
    )


def notebook_sample(frame, seed):
    positive = frame.loc[frame.purchase == 1]
    negative = frame.loc[frame.purchase == 0]
    return pd.concat([
        positive.sample(n=min(len(positive), 5000), random_state=seed),
        negative.sample(n=min(len(negative), 5000), random_state=seed + 13),
    ]).sample(frac=1, random_state=seed + 29).reset_index(drop=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {split: WORK / f"cosmetics_seed42_{split}.csv" for split in SPLIT_NAMES}
    splits = {split: read_sessions(path) for split, path in paths.items()}
    ids = {}
    users = {}
    for split, frame in splits.items():
        if frame.session_id_hash.duplicated().any() or frame.user_id.isna().any():
            raise ValueError(f"Unusable session or user IDs in {split}")
        ids[split] = set(frame.session_id_hash)
        users[split] = set(frame.user_id)
    overlap = [
        {"first_split": first, "second_split": second, "session_overlap": len(ids[first] & ids[second]),
         "user_overlap": len(users[first] & users[second])}
        for first, second in (("train", "val"), ("train", "test"), ("val", "test"))
    ]
    pd.DataFrame(overlap).to_csv(OUT / "split_overlap.csv", index=False)
    if any(row["session_overlap"] or row["user_overlap"] for row in overlap):
        raise ValueError("Cosmetics seed-42 splits contain session/user overlap")

    basis_path = WORK / "cosmetics_BP_basis_clean.json"
    basis = json.loads(basis_path.read_text(encoding="utf-8"))
    props = set(basis["vocab"])
    fit = notebook_sample(splits["train"], 42)
    select = notebook_sample(splits["val"], 42 + 99)
    sample_rows = []
    for split, sample, sample_name in (("train", fit, "train_sample"), ("val", select, "eval_sample")):
        saved_path = WORK / f"cosmetics_bp_{sample_name}.csv"
        saved = read_sessions(saved_path)
        trace_path = WORK / f"cosmetics_bp_{split}_json.json"
        original_traces = json.loads(trace_path.read_text(encoding="utf-8"))
        actual_trace_matches = {}
        for label, key in ((1, "traces_pos"), (0, "traces_neg")):
            reconstructed = [
                [sorted(state & props) for state in parse_event_trace(value)[-20:]]
                for value in sample.loc[sample.purchase == label, "prop_BDP"]
            ]
            actual_trace_matches[label] = reconstructed == original_traces[key]
        sample_rows.append({
            "formula_stage": "fit" if split == "train" else "select",
            "source_split": split, "n": len(sample), "positive": int(sample.purchase.sum()),
            "ordered_sample_csv_ids_match": sample.session_id_hash.tolist() == saved.session_id_hash.tolist(),
            "sample_id_sets_match": set(sample.session_id_hash) == set(saved.session_id_hash),
            "replayed_ids_found_in_saved_sample": len(set(sample.session_id_hash) & set(saved.session_id_hash)),
            "positive_traces_match": actual_trace_matches[1], "negative_traces_match": actual_trace_matches[0],
            "saved_sample_membership_in_expected_split": set(saved.session_id_hash) <= ids[split],
            "saved_sample_session_overlap_with_test": len(set(saved.session_id_hash) & ids["test"]),
            "saved_sample_user_overlap_with_test": len(set(saved.user_id) & users["test"]),
            "saved_sample_sha256": sha(saved_path), "saved_trace_json_sha256": sha(trace_path),
        })
    pd.DataFrame(sample_rows).to_csv(OUT / "formula_sample_audit.csv", index=False)
    records = basis["formulas"]
    trees = [ast.literal_eval(record["formula_tree"]) for record in records]
    val_data = json.loads((WORK / "cosmetics_bp_val_json.json").read_text(encoding="utf-8"))
    positive = np.stack([vector(trace, trees) for trace in val_data["traces_pos"]]).mean(axis=0)
    negative = np.stack([vector(trace, trees) for trace in val_data["traces_neg"]]).mean(axis=0)
    support = pd.DataFrame([
        {"formula": record["formula_str"], "saved_pos_support": record["pos_support"],
         "replayed_pos_support": positive[j], "saved_neg_support": record["neg_support"],
         "replayed_neg_support": negative[j]}
        for j, record in enumerate(records)
    ])
    support["matches"] = (
        np.isclose(support.saved_pos_support, support.replayed_pos_support, atol=1e-12, rtol=0)
        & np.isclose(support.saved_neg_support, support.replayed_neg_support, atol=1e-12, rtol=0)
    )
    support.to_csv(OUT / "basis_validation_support.csv", index=False)
    manifest = {
        "seed": 42, "basis_sha256": sha(basis_path),
        "split_sha256": {split: sha(path) for split, path in paths.items()},
        "split_counts": {split: len(frame) for split, frame in splits.items()},
        "split_positive_counts": {split: int(frame.purchase.sum()) for split, frame in splits.items()},
        "split_overlap": overlap, "formula_samples": sample_rows,
        "formula_support_pairs_matched": int(support.matches.sum()),
        "formula_support_pairs_total": len(records),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
