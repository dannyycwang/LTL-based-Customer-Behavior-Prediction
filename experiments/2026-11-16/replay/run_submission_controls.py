"""Run low-cost, matched Act+Rel controls and a fixed event-horizon experiment."""
from artifact_paths import (AUDIT_ROOT, WORK_DIR, MODEL_DIR, COVEO_DATA, COVEO_DIAGNOSTIC_DATA, COVEO_BASIS, ELECTRONICS_BASIS, COSMETICS_BASIS, resolve_saved_path)
import argparse
import ast
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from diagnose_coveo import freeze, sha, vector
from fix_cosmetics_parser import parse_event_trace
from report_coveo_seed42 import tex_table
from run_act_rel_seed42 import DEFAULT_BASIS, DEFAULT_REBUILD, PRE_PURCHASE_DATA, PRE_PURCHASE_OUT
from run_coveo import BASIS as COVEO_BASIS, DATA as COVEO_DATA, PROPS, ROOT, WORK
from verify_electronics_rebuild import DWELL_ATOMS

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, matthews_corrcoef
from sklearn.preprocessing import OneHotEncoder, StandardScaler


OUT = ROOT / "submission_controls_seed42"
SPLITS = ("train", "val", "test")
HORIZON = 10
KEYS = ["session_id_hash", "purchase"]
DATASETS = ("Coveo", "Electronics", "Cosmetics")


def audited_inputs(dataset):
    if dataset == "Coveo":
        run = ROOT / "runs" / "protected_first_v1_seed42_short_raw"
        meta = json.loads((run / "config.json").read_text())
        if sha(COVEO_DATA) != meta["dataset_sha256"] or sha(COVEO_BASIS) != meta["basis_sha256"]:
            raise ValueError("Coveo source hash mismatch")
        cols = ["action_sequence", "product_sku_hash_sequence", "category_hash_sequence"]
        source = pd.read_csv(COVEO_DATA, usecols=[*KEYS, *cols, "logic_trace_compact"],
                             dtype={"session_id_hash": str}).set_index("session_id_hash", verify_integrity=True)
        frames = {}
        for split in SPLITS:
            path = run / f"{split}_ids.csv"
            if sha(path) != meta["splits"][split]:
                raise ValueError("Coveo split hash mismatch")
            ids = pd.read_csv(path, dtype={"session_id": str})
            frame = source.loc[ids.session_id].reset_index()
            if not np.array_equal(frame.purchase, ids.label):
                raise ValueError("Coveo label replay mismatch")
            frames[split] = frame
        parse_trace = lambda value: [set(state) & set(PROPS) for state in json.loads(value)]
        return frames, cols, json.loads, "logic_trace_compact", parse_trace, COVEO_BASIS, PROPS, meta["splits"]

    if dataset == "Electronics":
        meta = json.loads((ROOT / "electronics_rebuild" / "seed42_feature_manifest.json").read_text())
        base = DEFAULT_REBUILD / "electronics_ready_noleak" / "electronics_splits" / "seed_42"
        cols = ["behavior_sequence", "product_sku_hash_sequence", "category_hash_sequence"]
        trace_col = "atom_sequence"
        parse_raw = json.loads
        parse_trace = lambda value: [set(state) - DWELL_ATOMS for state in json.loads(value)]
        basis = DEFAULT_BASIS
        hashes = {split: meta["split_hashes"][f"42_{split}"] for split in SPLITS}
        scaler = resolve_saved_path(meta["feature_dir"]) / "counts_scaler.npz"
    else:
        meta = json.loads((PRE_PURCHASE_OUT / "feature_manifest.json").read_text())
        base = PRE_PURCHASE_DATA
        cols = ["behavior_sequence", "product_sequence", "category_sequence"]
        trace_col = "prop_BP"
        parse_raw = str.split
        parse_trace = parse_event_trace
        basis = COSMETICS_BASIS
        hashes = meta["split_sha256"]
        scaler = base / "features" / "counts_scaler.npz"
    if sha(basis) != meta["basis_sha256"]:
        raise ValueError("Formula basis hash mismatch")
    frames = {}
    for split in SPLITS:
        path = base / f"{split}.csv"
        if sha(path) != hashes[split]:
            raise ValueError(f"Split hash mismatch: {dataset}/{split}")
        frames[split] = pd.read_csv(path, usecols=[*KEYS, *cols, trace_col, "user_id"],
                                    dtype={"session_id_hash": str, "user_id": str})
    with np.load(scaler) as archive:
        props = archive["propositions"][:-1].tolist()
    return frames, cols, parse_raw, trace_col, parse_trace, basis, props, hashes


def counts_and_prefix(frames, cols, parse_raw, trace_col, parse_trace, basis, props, with_transitions=False):
    records = json.loads(basis.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = records["formulas"]
    records = records[:10]
    trees = [ast.literal_eval(row["formula_tree"]) if isinstance(row["formula_tree"], str)
             else freeze(row["formula_tree"]) for row in records]
    if len(trees) != 10:
        raise ValueError("Expected the fixed K=10 basis")
    actions = [i for i, name in enumerate(props) if name in {"b", "d", "a", "r"}]
    prop_index = {atom: i for i, atom in enumerate(props)}
    data = {}
    for split, frame in frames.items():
        n = len(frame)
        # Right-padded positional tokens; both encoders see the same observed prefix.
        raw = np.full((n, 30), "<PAD>", dtype=object)
        length = np.empty(n, dtype=np.int32)
        counts = np.empty((n, len(props)), dtype=np.float64)
        prefix_counts = np.empty_like(counts)
        prefix_formula = np.empty((n, len(trees)), dtype=np.float64)
        full_formula = np.empty_like(prefix_formula)
        if with_transitions:
            transition_full = np.zeros((n, len(props) ** 2), dtype=bool)
            transition_prefix = np.zeros_like(transition_full)
        for i, row in enumerate(frame[[*cols, trace_col]].itertuples(index=False, name=None)):
            fields = [parse_raw(value) for value in row[:3]]
            trace = parse_trace(row[3])
            length[i] = len(trace)
            if length[i] < 1 or any(len(field) != length[i] for field in fields):
                raise ValueError(f"Raw/symbolic length mismatch at {split}/{i}")
            for j, field in enumerate(fields):
                tokens = list(map(str, field[:10]))
                raw[i, j * 10:j * 10 + len(tokens)] = tokens
            prefix = trace[:10]
            counts[i] = [sum(atom in state for state in trace) for atom in props]
            prefix_counts[i] = [sum(atom in state for state in prefix) for atom in props]
            prefix_formula[i] = vector(prefix, trees)
            full_formula[i] = vector(trace, trees)
            if with_transitions:
                encoded = [[prop_index[atom] for atom in state if atom in prop_index] for state in trace]
                observed_pairs = set()
                for pos, (left, right) in enumerate(zip(encoded, encoded[1:]), 1):
                    observed_pairs.update(a * len(props) + b for a in left for b in right)
                    if pos == HORIZON - 1:
                        transition_prefix[i, list(observed_pairs)] = True
                transition_full[i, list(observed_pairs)] = True
                if length[i] < HORIZON:
                    transition_prefix[i] = transition_full[i]
        if not np.array_equal(counts[:, actions].sum(axis=1), length):
            raise ValueError(f"Action counts do not reconstruct length for {split}")
        if not np.array_equal(prefix_counts[:, actions].sum(axis=1), np.minimum(length, 10)):
            raise ValueError(f"Prefix action counts do not reconstruct observed length for {split}")
        if not np.array_equal(prefix_formula[length <= 10], full_formula[length <= 10]):
            raise ValueError("Unexpected formula mismatch on short histories")
        data[split] = dict(raw=raw, length=length, counts=counts, prefix_counts=prefix_counts,
                           prefix_formula=prefix_formula, full_formula=full_formula,
                           session_id=frame.session_id_hash.to_numpy(), label=frame.purchase.to_numpy())
        if "user_id" in frame:
            data[split]["user_id"] = frame.user_id.to_numpy()
        if with_transitions:
            if not np.array_equal(transition_prefix[length <= HORIZON], transition_full[length <= HORIZON]):
                raise ValueError("Prefix and full transitions differ on short histories")
            data[split].update(transition_full=transition_full, transition_prefix=transition_prefix)
    return data, actions, [row["formula_str"] for row in records]


def verify_test_replay(dataset, test):
    name = dataset.lower() if dataset != "Cosmetics" else "cosmetics_pre_purchase"
    directory = ROOT / f"{name}_seed42_comparison"
    with np.load(directory / "paired_satisfaction.npz") as saved:
        for existing, computed in (
            ("session_id", "session_id"), ("label", "label"), ("length", "length"),
            ("prefix", "prefix_formula"), ("full", "full_formula"),
        ):
            if not np.array_equal(saved[existing], test[computed]):
                raise ValueError(f"Prepared control features disagree with audited test {existing}: {dataset}")


def cohorts(data, mode, dataset):
    result, rows = {}, []
    for split, table in data.items():
        if mode == "full_observation":
            keep = np.ones(len(table["length"]), dtype=bool)
            target = table["label"]
        else:
            # A positive's retained trace ends immediately before the first purchase.
            keep = table["length"] >= HORIZON
            target = ((table["label"] == 1) & (table["length"] < 2 * HORIZON)).astype(int)
        result[split] = {key: value[keep] for key, value in table.items()}
        result[split]["label"] = target[keep]
        if len(np.unique(result[split]["label"])) != 2:
            raise ValueError(f"{dataset}/{split} lacks both classes in {mode}")
        rows.append(dict(dataset=dataset, protocol=mode, split=split, n=int(keep.sum()),
                         positive=int(target[keep].sum()), positive_rate=float(target[keep].mean()),
                         excluded=int((~keep).sum())))
    return result, rows


def features(table, mode, actions):
    length = table["length"][:, None].astype(float)
    if mode == "fixed_event_horizon":
        return {
            "Short raw": None,
            "Short + action counts": table["prefix_counts"][:, actions],
            "Short + Act+Rel counts": table["prefix_counts"],
            "Short + Prefix-LTLf": table["prefix_formula"],
        }
    counts = table["counts"]
    action_counts = counts[:, actions]
    proportions = action_counts / length
    return {
        "Length only": np.log1p(length),
        "Short raw": None,
        "Short + length": np.log1p(length),
        "Short + action proportions": proportions,
        "Short + action counts": action_counts,
        "Short + Act+Rel counts": counts,
        "Short + Act+Rel counts and length": np.column_stack([counts, length]),
        "Short + Prefix-LTLf": table["prefix_formula"],
        "Short + Full-LTLf": table["full_formula"],
    }


def fit_controls(dataset, data, mode, actions, out, numerical=None):
    encoder = OneHotEncoder(handle_unknown="ignore")
    x_raw = {"train": encoder.fit_transform(data["train"]["raw"])}
    for split in ("val", "test"):
        x_raw[split] = encoder.transform(data[split]["raw"])
    if numerical is None:
        numerical = {split: features(values, mode, actions) for split, values in data.items()}
    rows, predictions = [], {}
    for name, train_values in numerical["train"].items():
        if train_values is None:
            x = x_raw
        else:
            scaler = StandardScaler().fit(train_values)
            x = {}
            for split in SPLITS:
                z = csr_matrix(scaler.transform(numerical[split][name]))
                x[split] = z if name == "Length only" else hstack([x_raw[split], z], format="csr")
        model = LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, random_state=42)
        start = time.perf_counter()
        model.fit(x["train"], data["train"]["label"])
        train_seconds = time.perf_counter() - start
        if model.n_iter_[0] >= model.max_iter:
            raise RuntimeError(f"Logistic regression did not converge: {dataset}/{mode}/{name}")
        p_val = model.predict_proba(x["val"])[:, 1]
        p_test = model.predict_proba(x["test"])[:, 1]
        y_val, y_test = data["val"]["label"], data["test"]["label"]
        candidates = np.r_[.001, np.arange(.01, 1, .01), .999]
        threshold = max(candidates, key=lambda t: matthews_corrcoef(y_val, p_val >= t))
        predicted = p_test >= threshold
        rows.append(dict(dataset=dataset, protocol=mode, condition=name, n_test=len(y_test),
                         positives=int(y_test.sum()), threshold=threshold,
                         MCC=matthews_corrcoef(y_test, predicted), AUPRC=average_precision_score(y_test, p_test),
                         F1=f1_score(y_test, predicted, zero_division=0), Brier=brier_score_loss(y_test, p_test),
                         train_seconds=train_seconds, iterations=int(model.n_iter_[0])))
        predictions[name] = p_test
        print(f"{dataset} / {mode} / {name}: {rows[-1]['MCC']:.4f} MCC", flush=True)
    test = data["test"]
    pd.DataFrame(dict(session_id=test["session_id"], label=test["label"], **predictions)).to_csv(
        out / f"{dataset.lower()}_{mode}_predictions.csv.gz", index=False
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    metrics, sizes = [], []
    for dataset in (args.dataset,) if args.dataset else DATASETS:
        frames, cols, parse_raw, trace_col, parse_trace, basis, props, hashes = audited_inputs(dataset)
        tables, action_indices, formulas = counts_and_prefix(frames, cols, parse_raw, trace_col, parse_trace, basis, props)
        verify_test_replay(dataset, tables["test"])
        metadata = dict(dataset=dataset, split_seed=42, k=len(formulas), formula_basis_sha256=sha(basis),
                        formulas=formulas, feature_props=props, split_hashes=hashes, fixed_observation_events=10,
                        fixed_target_next_events=10)
        (OUT / f"{dataset.lower()}_provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if "user_id" in tables["test"]:
            pd.DataFrame({"session_id": tables["test"]["session_id"], "user_id": tables["test"]["user_id"]}).to_csv(
                WORK / f"{dataset.lower()}_bootstrap_clusters.csv", index=False
            )
        for mode in ("full_observation", "fixed_event_horizon"):
            cohort, rows = cohorts(tables, mode, dataset)
            sizes.extend(rows)
            metrics.extend(fit_controls(dataset, cohort, mode, action_indices, OUT))
    prefix = f"{args.dataset.lower()}_" if args.dataset else ""
    scores = pd.DataFrame(metrics)
    scores.to_csv(OUT / f"{prefix}metrics.csv", index=False)
    pd.DataFrame(sizes).to_csv(OUT / f"{prefix}cohort_counts.csv", index=False)
    for protocol, frame in scores.groupby("protocol", sort=False):
        tex_table(frame[["dataset", "condition", "n_test", "MCC", "AUPRC", "F1", "Brier"]],
                  OUT / f"{prefix}{protocol}.tex", ".4f")


if __name__ == "__main__":
    main()
