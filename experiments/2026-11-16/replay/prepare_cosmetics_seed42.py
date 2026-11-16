"""Prepare corrected BP inputs for the historical Cosmetics observation boundary."""
from artifact_paths import (AUDIT_ROOT, WORK_DIR, MODEL_DIR, COVEO_DATA, COVEO_DIAGNOSTIC_DATA, COVEO_BASIS, ELECTRONICS_BASIS, COSMETICS_BASIS, resolve_saved_path)
import ast
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from diagnose_coveo import sha, validate, vector
from fix_cosmetics_parser import parse_event_trace


ROOT = AUDIT_ROOT
WORK = WORK_DIR
AUDIT = ROOT / "cosmetics_seed42_audit"
FEATURES = WORK / "cosmetics_seed42_act_rel_features"


def prepare_features(audit, paths, features, manifest_path, observation_protocol):
    if not all(row["positive_traces_match"] and row["negative_traces_match"]
               for row in audit["formula_samples"]):
        raise ValueError("Formula sample provenance check failed")
    if audit["formula_support_pairs_matched"] != audit["formula_support_pairs_total"]:
        raise ValueError("Formula validation support check failed")
    basis_path = COSMETICS_BASIS
    if sha(basis_path) != audit["basis_sha256"]:
        raise ValueError("Basis changed since audit")
    basis = json.loads(basis_path.read_text(encoding="utf-8"))
    records = basis["formulas"][:10]
    names = [record["formula_str"] for record in records]
    trees = [ast.literal_eval(record["formula_tree"]) for record in records]
    semantic_checks = validate(trees)
    props = basis["vocab"]
    frames = []
    for split, path in paths.items():
        if sha(path) != audit["split_sha256"][split]:
            raise ValueError(f"Split changed since audit: {split}")
        frames.append(pd.read_csv(
            path, usecols=["session_id_hash", "purchase", "n_events", "prop_BP",
                           "behavior_sequence", "product_sequence", "category_sequence"],
            dtype={"session_id_hash": str},
        ).assign(split=split))
    data = pd.concat(frames, ignore_index=True)
    traces = [parse_event_trace(value) for value in data.prop_BP]
    lengths = np.array([len(trace) for trace in traces])
    if not np.array_equal(lengths, data.n_events.to_numpy()):
        raise ValueError("Symbolic trace lengths differ from recorded event lengths")
    if set().union(*(state for trace in traces for state in trace)) - set(props):
        raise ValueError("Unexpected BP propositions")
    for raw_column in ("behavior_sequence", "product_sequence", "category_sequence"):
        if not np.array_equal(data[raw_column].str.split().str.len().to_numpy(), lengths):
            raise ValueError(f"Raw field lengths differ from BP trace: {raw_column}")
    start = time.perf_counter()
    prefix = np.stack([vector(trace[:10], trees) for trace in traces])
    prefix_seconds = time.perf_counter() - start
    start = time.perf_counter()
    full = np.stack([vector(trace, trees) for trace in traces])
    full_seconds = time.perf_counter() - start
    if not np.array_equal(prefix[lengths <= 10], full[lengths <= 10]):
        raise ValueError("Prefix/full mismatch at L<=10")
    train = data.split.to_numpy() == "train"
    start = time.perf_counter()
    counts = np.array([[sum(atom in state for state in trace) for atom in props] + [len(trace)]
                       for trace in traces], dtype=np.float32)
    mean = counts[train].mean(axis=0)
    std = counts[train].std(axis=0)
    std[std == 0] = 1
    counts = (counts - mean) / std
    counts_seconds = time.perf_counter() - start
    files = {}
    for key, matrix, columns in (
        ("prefix_ltlf", prefix, names), ("full_ltlf", full, names), ("counts", counts, [*props, "trace_length"])
    ):
        directory = features / key
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"Z_BP_top{matrix.shape[1]}_cosmetics_dec.npz"
        np.savez_compressed(path, Z=matrix, y=data.purchase.to_numpy(),
                            session_id_hash=data.session_id_hash.to_numpy(str), formulas=np.array(columns))
        files[key] = {"path": str(path), "sha256": sha(path), "dimension": matrix.shape[1]}
    np.savez(features / "counts_scaler.npz", mean=mean, std=std, propositions=np.array([*props, "trace_length"]))
    metadata = {
        "seed": 42, "basis_sha256": audit["basis_sha256"], "split_sha256": audit["split_sha256"],
        "semantic_checks": semantic_checks, "features": files,
        "test_n": int((data.split == "test").sum()),
        "observation_protocol": observation_protocol,
        "feature_construction_seconds_excluding_parsing": {
            "prefix_ltlf": prefix_seconds, "full_ltlf": full_seconds, "counts": counts_seconds,
        },
    }
    manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    prepare_features(
        json.loads((AUDIT / "manifest.json").read_text(encoding="utf-8")),
        {split: WORK / f"cosmetics_seed42_{split}.csv" for split in ("train", "val", "test")},
        FEATURES,
        AUDIT / "feature_manifest.json",
        "Historical purchase-masked session with events retained after purchase",
    )
