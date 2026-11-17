"""Replay saved feature matrices and produce seed-42 formula diagnostics."""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from diagnose_coveo import sha, vector
from fix_cosmetics_parser import parse_event_trace
from prepare_electronics_seed42 import DWELL_ATOMS
from report_coveo_seed42 import draw_formula_diagnostics
from run_act_rel_seed42 import (
    DEFAULT_BASIS, DEFAULT_REBUILD, PRE_PURCHASE_DATA, PRE_PURCHASE_OUT, ROOT, WORK,
)


def load_test(dataset):
    if dataset == "electronics":
        feature_meta = ROOT / "electronics_rebuild" / "seed42_feature_manifest.json"
        meta = json.loads(feature_meta.read_text(encoding="utf-8"))
        path = DEFAULT_REBUILD / "electronics_ready_noleak" / "electronics_splits" / "seed_42" / "test.csv"
        expected = meta["split_hashes"]["42_test"]
        features = {key: (Path(meta["feature_dir"]) / f"{key}.npz", meta["feature_hashes"][f"{key}.npz"])
                    for key in ("prefix_ltlf", "counts", "full_ltlf")}
        scaler = Path(meta["feature_dir"]) / "counts_scaler.npz"
        basis_path = DEFAULT_BASIS
        trace_column, length_column = "atom_sequence", "session_len"
        parse = lambda value: [set(state) - DWELL_ATOMS for state in json.loads(value)]
    else:
        pre_purchase = dataset == "cosmetics_pre_purchase"
        audit = PRE_PURCHASE_OUT if pre_purchase else ROOT / "cosmetics_seed42_audit"
        feature_meta = audit / "feature_manifest.json"
        meta = json.loads(feature_meta.read_text(encoding="utf-8"))
        path = PRE_PURCHASE_DATA / "test.csv" if pre_purchase else WORK / "cosmetics_seed42_test.csv"
        expected = meta["split_sha256"]["test"]
        features = {key: (Path(value["path"]), value["sha256"]) for key, value in meta["features"].items()}
        feature_dir = PRE_PURCHASE_DATA / "features" if pre_purchase else WORK / "cosmetics_seed42_act_rel_features"
        scaler = feature_dir / "counts_scaler.npz"
        basis_path = WORK / "cosmetics_BP_basis_clean.json"
        trace_column, length_column = "prop_BP", "n_events"
        parse = parse_event_trace
    if sha(path) != expected or sha(basis_path) != meta["basis_sha256"]:
        raise ValueError("Data or basis differs from the audited inputs")
    test = pd.read_csv(path, dtype={"session_id_hash": str},
                       usecols=["session_id_hash", "purchase", length_column, trace_column])
    records = json.loads(basis_path.read_text(encoding="utf-8"))
    records = (records if dataset == "electronics" else records["formulas"])[:10]
    trees = [ast.literal_eval(record["formula_tree"]) for record in records]
    start = time.perf_counter()
    traces = [parse(value) for value in test[trace_column]]
    parse_sec = time.perf_counter() - start
    lengths = np.array([len(trace) for trace in traces])
    if not np.array_equal(lengths, test[length_column].to_numpy()):
        raise ValueError("Unexpected trace lengths")
    start = time.perf_counter()
    prefix = np.stack([vector(trace[:10], trees) for trace in traces])
    prefix_sec = time.perf_counter() - start
    start = time.perf_counter()
    full = np.stack([vector(trace, trees) for trace in traces])
    full_sec = time.perf_counter() - start
    if not np.array_equal(prefix[lengths <= 10], full[lengths <= 10]):
        raise ValueError("Prefix and full differ at L<=10")
    with np.load(scaler) as archive:
        propositions = archive["propositions"][:-1].tolist()
        start = time.perf_counter()
        counts = np.array([[sum(atom in state for state in trace) for atom in propositions] + [len(trace)]
                           for trace in traces], dtype=np.float32)
        counts = (counts - archive["mean"]) / archive["std"]
        counts_sec = time.perf_counter() - start
    for key, values in (("prefix_ltlf", prefix), ("counts", counts), ("full_ltlf", full)):
        archive_path, digest = features[key]
        if sha(archive_path) != digest:
            raise ValueError(f"Unexpected feature archive for {key}")
        with np.load(archive_path) as archive:
            n = len(test)
            if not (np.array_equal(archive["session_id_hash"][-n:], test.session_id_hash.to_numpy(str))
                    and np.array_equal(archive["y"][-n:], test.purchase.to_numpy())
                    and np.array_equal(archive["Z"][-n:], values)):
                raise ValueError(f"Replayed test features differ for {key}")
    timing = dict(symbolic_parse_sec=parse_sec, prefix_ltlf=prefix_sec, counts=counts_sec, full_ltlf=full_sec)
    return test, records, trees, traces, lengths, prefix, full, timing


def diagnose(dataset):
    test, records, trees, traces, lengths, prefix, full, timing = load_test(dataset)
    out = ROOT / f"{dataset}_seed42_comparison"
    out.mkdir(exist_ok=True)
    ids = [record["formula_id"] for record in records]
    rows = []
    for group, mask in (("all", lengths > 0), ("L<=10", lengths <= 10), ("L>10", lengths > 10)):
        a, b = prefix[mask], full[mask]
        changed = a != b
        for column, record in enumerate(records):
            rows.append(dict(stratum=group, formula_id=ids[column], formula=record["formula_str"],
                             n=len(a), changed_rate=float(changed[:, column].mean()),
                             zero_to_one_rate=float(((a == 0) & (b == 1))[:, column].mean()),
                             one_to_zero_rate=float(((a == 1) & (b == 0))[:, column].mean())))
        rows.append(dict(stratum=group, formula_id="ANY_BIT", n=len(a),
                         changed_rate=float(changed.any(axis=1).mean())))
    pd.DataFrame(rows).to_csv(out / "formula_changes.csv", index=False)
    np.savez_compressed(out / "paired_satisfaction.npz", session_id=test.session_id_hash.to_numpy(str),
                        label=test.purchase.to_numpy(), length=lengths, prefix=prefix, full=full)

    seeds = list(range(202611100, 202611110))
    shuffled_rows = []
    state_counts = [Counter(map(frozenset, trace)) for trace in traces]
    for seed in seeds:
        rng = np.random.default_rng(seed)
        replay = np.empty_like(full)
        for index, trace in enumerate(traces):
            shuffled = [trace[j] for j in rng.permutation(len(trace))]
            if Counter(map(frozenset, shuffled)) != state_counts[index]:
                raise ValueError("Shuffle changed proposition counts or co-occurrence")
            replay[index] = vector(shuffled, trees)
        changed = replay != full
        rates = [*changed.mean(axis=0), changed.any(axis=1).mean()]
        shuffled_rows.extend(dict(shuffle_seed=seed, formula_id=formula, n=len(test), changed_rate=rate)
                             for formula, rate in zip([*ids, "ANY_BIT"], rates))
    pd.DataFrame(shuffled_rows).to_csv(out / "shuffle_changes.csv", index=False)

    candidates = [(int(lengths[i]), test.session_id_hash.iloc[i], int(j), int(i))
                  for i, j in zip(*np.where((prefix == 0) & (full == 1) & (lengths[:, None] > 10)))]
    if candidates:
        length, sid, column, index = min(candidates)
        trace, tree = traces[index], trees[column]
        horizons = [int(vector(trace[:end], [tree])[0]) for end in range(10, length + 1)]
        first_true = horizons.index(1) + 10
        case = dict(selection_rule="Shortest false-to-true trace, then session ID and basis order.",
                    session_id=sid, length=length, formula_id=ids[column],
                    formula=records[column]["formula_str"], prefix_value=0, full_value=1,
                    first_true_length=first_true, satisfaction_by_horizon_10_onward=horizons,
                    endpoint_propositions=sorted(trace[first_true - 1]),
                    trace=[sorted(state) for state in trace])
        (out / "real_trace_case.json").write_text(json.dumps(case, indent=2), encoding="utf-8")
    meta = dict(seed=42, n=len(test), shuffle_seeds=seeds, features_match_audited_archives=True,
                test_feature_timing=timing,
                basis=[record["formula_str"] for record in records])
    (out / "formula_diagnostic_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    draw_formula_diagnostics(out)
    print(dataset, pd.DataFrame(rows).query("formula_id == 'ANY_BIT'").to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("electronics", "cosmetics", "cosmetics_pre_purchase"))
    args = parser.parse_args()
    for dataset in [args.dataset] if args.dataset else ("electronics", "cosmetics"):
        diagnose(dataset)
