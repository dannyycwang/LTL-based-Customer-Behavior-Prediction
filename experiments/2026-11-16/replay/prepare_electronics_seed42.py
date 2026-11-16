"""Prepare verified seed-42 Electronics inputs without training a predictor."""
from artifact_paths import (AUDIT_ROOT, WORK_DIR, MODEL_DIR, COVEO_DATA, COVEO_DIAGNOSTIC_DATA, COVEO_BASIS, ELECTRONICS_BASIS, COSMETICS_BASIS, resolve_saved_path)
import ast
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from diagnose_coveo import sha, validate, vector
from verify_electronics_rebuild import DEFAULT_BASIS, DEFAULT_REBUILD, DWELL_ATOMS, load_split


ROOT = AUDIT_ROOT
AUDIT = ROOT / "electronics_rebuild"
READY = DEFAULT_REBUILD / "electronics_ready_noleak"
FEATURES = DEFAULT_REBUILD / "seed42_act_rel_features"


def main():
    source = json.loads((AUDIT / "manifest.json").read_text(encoding="utf-8"))
    if source["formula_support_pairs_matched"] != 10 or sha(DEFAULT_BASIS) != source["basis_sha256"]:
        raise ValueError("Expected the audited, unchanged seed-42 BP basis")
    records = json.loads(DEFAULT_BASIS.read_text(encoding="utf-8"))
    trees = [ast.literal_eval(record["formula_tree"]) for record in records]
    semantic_checks = validate(trees)
    props = source["formula_pool_vocabulary"]
    frames = []
    for split in ("train", "val", "test"):
        split_path = READY / "electronics_splits" / "seed_42" / f"{split}.csv"
        if sha(split_path) != source["split_hashes"][f"42_{split}"]:
            raise ValueError(f"Split hash mismatch for {split}")
        frames.append(load_split(READY, 42, split).assign(split=split))
    data = pd.concat(frames, ignore_index=True)
    traces = [[set(state) - DWELL_ATOMS for state in json.loads(value)] for value in data.atom_sequence]
    observed = set().union(*(state for trace in traces for state in trace))
    if observed != set(props):
        raise ValueError(f"Observed Act+Rel vocabulary differs from the audited BP basis: {observed}")
    lengths = np.array([len(trace) for trace in traces])
    if not np.array_equal(lengths, data.session_len.to_numpy()):
        raise ValueError("Trace lengths do not match raw session lengths")
    if any(len(json.loads(action)) != length or len(json.loads(product)) != length
           or len(json.loads(category)) != length
           for action, product, category, length in zip(
               data.action_sequence, data.product_sku_hash_sequence, data.category_hash_sequence, lengths
           )):
        raise ValueError("Raw APC sequences and symbolic traces are misaligned")
    pool = pd.read_csv(
        DEFAULT_REBUILD / "formula_learning_noleak_4000_last10_seed42" / "BP" / "formula_pool_ids.csv",
        dtype={"session_id_hash": str},
    )
    train = data.split.to_numpy() == "train"
    if not set(pool.session_id_hash) <= set(data.loc[train, "session_id_hash"]):
        raise ValueError("Formula pool is not wholly contained in the training set")
    start = time.perf_counter()
    prefix = np.stack([vector(trace[:10], trees) for trace in traces])
    prefix_seconds = time.perf_counter() - start
    start = time.perf_counter()
    full = np.stack([vector(trace, trees) for trace in traces])
    full_seconds = time.perf_counter() - start
    if not np.array_equal(prefix[lengths <= 10], full[lengths <= 10]):
        raise ValueError("Prefix/full mismatch at L<=10")
    start = time.perf_counter()
    counts = np.array([[sum(atom in state for state in trace) for atom in props] + [len(trace)]
                       for trace in traces], dtype=np.float32)
    mean = counts[train].mean(axis=0)
    std = counts[train].std(axis=0)
    std[std == 0] = 1
    counts = (counts - mean) / std
    counts_seconds = time.perf_counter() - start
    FEATURES.mkdir(parents=True, exist_ok=True)
    names = [record["formula_str"] for record in records]
    test = data.split.to_numpy() == "test"
    for key, z, columns in (
        ("prefix_ltlf", prefix, names), ("full_ltlf", full, names), ("counts", counts, [*props, "trace_length"])
    ):
        np.savez_compressed(FEATURES / f"{key}.npz", Z=z, y=data.purchase.to_numpy(),
                            session_id_hash=data.session_id_hash.to_numpy(str), formulas=np.array(columns))
    np.savez(FEATURES / "counts_scaler.npz", mean=mean, std=std, propositions=np.array([*props, "trace_length"]))
    rows = []
    for stratum, mask in (("all", test), ("L<=10", test & (lengths <= 10)), ("L>10", test & (lengths > 10))):
        a, b = prefix[mask], full[mask]
        for j, record in enumerate(records):
            rows.append({"stratum": stratum, "formula_id": record["formula_id"], "formula": names[j],
                         "n": int(mask.sum()), "positive_rate": float(data.loc[mask, "purchase"].mean()),
                         "changed_rate": float((a[:, j] != b[:, j]).mean()),
                         "zero_to_one_rate": float(((a[:, j] == 0) & (b[:, j] == 1)).mean()),
                         "one_to_zero_rate": float(((a[:, j] == 1) & (b[:, j] == 0)).mean())})
        rows.append({"stratum": stratum, "formula_id": "ANY_BIT", "n": int(mask.sum()),
                     "positive_rate": float(data.loc[mask, "purchase"].mean()),
                     "changed_rate": float((a != b).any(axis=1).mean())})
    pd.DataFrame(rows).to_csv(AUDIT / "seed42_formula_scope.csv", index=False)
    meta = {
        "seed": 42, "basis_sha256": source["basis_sha256"],
        "split_hashes": {key: value for key, value in source["split_hashes"].items() if key.startswith("42_")},
        "semantic_checks": semantic_checks, "feature_dir": str(FEATURES),
        "feature_construction_seconds_excluding_parsing": {
            "prefix_ltlf": prefix_seconds, "full_ltlf": full_seconds, "counts": counts_seconds,
        },
        "feature_hashes": {path.name: sha(path) for path in FEATURES.glob("*.npz")},
    }
    (AUDIT / "seed42_feature_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).query("formula_id == 'ANY_BIT'").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
