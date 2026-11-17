"""Evaluate formula scope and shuffle sensitivity on the protected seed-42 test set."""
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from diagnose_coveo import freeze, sha, validate, vector


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
OUT = ROOT / "coveo_seed42_comparison"
SOURCE = Path("C:/Users/ycw/Downloads/browsing_draw_random4_sessions_product_compact.csv")
PROPOSITIONS = ("b", "d", "a", "r", "fp", "rp", "sp", "pc", "p2", "p3", "sc", "cc", "rc")
ATOMS = set(PROPOSITIONS)


def main():
    prefix_run = RUNS / "protected_first_v1_seed42_prefix_ltlf"
    full_run = RUNS / "protected_first_v1_seed42_full_ltlf"
    config = json.loads((prefix_run / "config.json").read_text(encoding="utf-8"))
    if sha(SOURCE) != config["dataset_sha256"]:
        raise ValueError("Source differs from training config")
    test_path = prefix_run / "test_ids.csv"
    if sha(test_path) != config["splits"]["test"]:
        raise ValueError("Test IDs differ from training config")
    test = pd.read_csv(test_path, dtype={"session_id": str})
    records = json.loads((prefix_run / "basis.json").read_text(encoding="utf-8"))
    if records != json.loads((full_run / "basis.json").read_text(encoding="utf-8")):
        raise ValueError("Prefix and full formula bases differ")
    trees = [freeze(record["formula_tree"]) for record in records]
    comparisons = validate(trees)
    test_set = set(test.session_id)
    traces = {}
    row_positions = {}
    labels = {}
    columns = ["session_id_hash", "logic_trace_compact", "action_sequence", "purchase"]
    parse_sec = 0.0
    for chunk in pd.read_csv(SOURCE, usecols=columns, chunksize=10_000,
                             dtype={"session_id_hash": str}):
        for row in chunk.loc[chunk.session_id_hash.isin(test_set)].itertuples():
            start = time.perf_counter()
            trace = [set(state) & ATOMS for state in json.loads(row.logic_trace_compact)]
            parse_sec += time.perf_counter() - start
            if row.session_id_hash in traces or len(trace) != len(json.loads(row.action_sequence)):
                raise ValueError("Duplicate session or trace length mismatch")
            traces[row.session_id_hash] = trace
            row_positions[row.session_id_hash] = row.Index
            labels[row.session_id_hash] = row.purchase
    ordered = [traces[session] for session in test.session_id]
    positions = [row_positions[session] for session in test.session_id]
    if [labels[session] for session in test.session_id] != test.label.tolist():
        raise ValueError("Source labels differ from training test IDs")
    start = time.perf_counter()
    prefix = np.stack([vector(trace[:10], trees) for trace in ordered])
    prefix_sec = time.perf_counter() - start
    start = time.perf_counter()
    full = np.stack([vector(trace, trees) for trace in ordered])
    full_sec = time.perf_counter() - start
    counts_run = RUNS / "protected_first_v1_seed42_counts"
    with np.load(counts_run / "counts_scaler_train_only.npz") as scaler:
        if tuple(scaler["propositions"]) != (*PROPOSITIONS, "trace_length"):
            raise ValueError("Unexpected Counts feature order")
        start = time.perf_counter()
        counts = np.array([[sum(atom in state for state in trace) for atom in PROPOSITIONS]
                           + [len(trace)] for trace in ordered], dtype=np.float32)
        counts = (counts - scaler["mean"]) / scaler["std"]
        counts_sec = time.perf_counter() - start
    if not np.array_equal(np.load(counts_run / "features.npy", mmap_mode="r")[positions], counts):
        raise ValueError("Replayed Counts features differ from training")
    for run, values in ((prefix_run, prefix), (full_run, full)):
        if not np.array_equal(np.load(run / "features.npy", mmap_mode="r")[positions], values):
            raise ValueError(f"Replayed features differ from {run.name}")
    lengths = np.array([len(trace) for trace in ordered])
    if not np.array_equal(prefix[lengths <= 10], full[lengths <= 10]):
        raise ValueError("Prefix/full mismatch on traces of at most 10 events")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for stratum, mask in (("all", lengths > 0), ("L<=10", lengths <= 10), ("L>10", lengths > 10)):
        a, b = prefix[mask], full[mask]
        for j, record in enumerate(records):
            rows.append({
                "stratum": stratum, "formula_id": record["formula_id"],
                "formula": record["formula_str"], "n": int(mask.sum()),
                "changed_rate": float((a[:, j] != b[:, j]).mean()),
                "zero_to_one_rate": float(((a[:, j] == 0) & (b[:, j] == 1)).mean()),
                "one_to_zero_rate": float(((a[:, j] == 1) & (b[:, j] == 0)).mean()),
            })
        rows.append({"stratum": stratum, "formula_id": "ANY_BIT", "n": int(mask.sum()),
                     "changed_rate": float((a != b).any(axis=1).mean())})
    pd.DataFrame(rows).to_csv(OUT / "formula_changes.csv", index=False)
    np.savez_compressed(OUT / "paired_satisfaction.npz", session_id=test.session_id.to_numpy(str),
                        label=test.label.to_numpy(), length=lengths, prefix=prefix, full=full)
    for j, record in enumerate(records):
        if "rp" not in record["formula_str"]:
            continue
        candidates = np.flatnonzero((lengths > 10) & (prefix[:, j] == 0) & (full[:, j] == 1))
        if not len(candidates):
            continue
        i = min(candidates, key=lambda index: (lengths[index], test.session_id.iloc[index]))
        trace = ordered[i]
        tree = trees[j]
        by_horizon = [int(vector(trace[:end], [tree])[0]) for end in range(10, len(trace) + 1)]
        first_true_length = 10 + by_horizon.index(1)
        proof_trace = trace[:first_true_length]
        witness = None
        if tree[0] == "U":
            left, right = tree[1:]
            for start in range(first_true_length):
                if vector(proof_trace[start:], [right])[0] and all(
                    vector(proof_trace[k:], [left])[0] for k in range(start)
                ):
                    witness = start + 1
                    break
        case = {
            "selection_rule": "First basis formula with rp and a false-to-true bit change. Choose the shortest test session and break ties by session ID.",
            "session_id": test.session_id.iloc[i], "length": len(trace), "formula_id": record["formula_id"],
            "formula": record["formula_str"], "prefix_value": 0, "full_value": 1,
            "first_true_length": first_true_length, "outer_until_witness_position": witness,
            "satisfaction_by_horizon_10_onward": by_horizon,
            "trace": [sorted(state) for state in trace],
        }
        (OUT / "real_trace_case.json").write_text(json.dumps(case, indent=2), encoding="utf-8")
        break
    seeds = list(range(202611100, 202611110))
    shuffle_rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        shuffled_values = np.empty_like(full)
        for i, trace in enumerate(ordered):
            shuffled = [trace[j] for j in rng.permutation(len(trace))]
            if Counter(map(frozenset, trace)) != Counter(map(frozenset, shuffled)):
                raise ValueError("Shuffle changed the proposition co-occurrence counts")
            shuffled_values[i] = vector(shuffled, trees)
        changed = shuffled_values != full
        for j, record in enumerate(records):
            shuffle_rows.append({"shuffle_seed": seed, "formula_id": record["formula_id"],
                                 "n": len(test), "changed_rate": float(changed[:, j].mean())})
        shuffle_rows.append({"shuffle_seed": seed, "formula_id": "ANY_BIT",
                             "n": len(test), "changed_rate": float(changed.any(axis=1).mean())})
    pd.DataFrame(shuffle_rows).to_csv(OUT / "shuffle_changes.csv", index=False)
    metadata = {"test_ids_sha256": sha(test_path), "dataset_sha256": config["dataset_sha256"],
                "basis_sha256": config["basis_sha256"], "shuffle_seeds": seeds,
                "reference_semantics_checks": comparisons, "n": len(test),
                "features_match_training": True,
                "test_symbolic_parse_sec": parse_sec,
                "test_feature_seconds_from_parsed_traces": {
                    "prefix_ltlf": prefix_sec, "counts": counts_sec, "full_ltlf": full_sec,
                }}
    (OUT / "formula_diagnostic_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(rows).query("formula_id == 'ANY_BIT'").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
