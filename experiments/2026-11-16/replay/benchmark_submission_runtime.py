"""Measure matched test inference from in-memory session fields to probabilities."""
from artifact_paths import (AUDIT_ROOT, WORK_DIR, MODEL_DIR, COVEO_DATA, COVEO_DIAGNOSTIC_DATA, COVEO_BASIS, ELECTRONICS_BASIS, COSMETICS_BASIS, resolve_saved_path)
import argparse
import ast
import json
from pathlib import Path
import platform
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from bootstrap_submission_controls import GROUPS
from diagnose_coveo import freeze, vector
from report_coveo_seed42 import tex_table
from run_act_rel_seed42 import PRE_PURCHASE_DATA, original, model_for, predict as secondary_predict
from run_coveo import load_originals, predict as coveo_predict, trim_full_batch
from run_submission_controls import audited_inputs, DATASETS, OUT, ROOT


CONDITIONS = {
    "full_raw": "Full raw",
    "short_raw": "Short raw",
    "prefix_ltlf": "Short + Prefix-LTLf",
    "counts": "Short + Counts",
    "full_ltlf": "Short + Full-LTLf",
}
SYMBOLIC = {"prefix_ltlf", "counts", "full_ltlf"}
DEVICE = torch.device("cuda")


def source_for(dataset, split="test"):
    frames, cols, parse_raw, trace_col, parse_trace, basis, props, _ = audited_inputs(dataset)
    test = frames[split]
    records = json.loads(basis.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = records["formulas"]
    trees = [ast.literal_eval(row["formula_tree"]) if isinstance(row["formula_tree"], str)
             else freeze(row["formula_tree"]) for row in records[:10]]
    if dataset == "Coveo":
        path = ROOT / "runs" / f"{GROUPS[dataset]}_counts" / "counts_scaler_train_only.npz"
    elif dataset == "Electronics":
        meta = json.loads((ROOT / "electronics_rebuild" / "seed42_feature_manifest.json").read_text())
        path = resolve_saved_path(meta["feature_dir"]) / "counts_scaler.npz"
    else:
        path = PRE_PURCHASE_DATA / "features" / "counts_scaler.npz"
    with np.load(path) as archive:
        if archive["propositions"][:-1].tolist() != list(props):
            raise ValueError("Counts vocabulary mismatch")
        center, scale = archive["mean"], archive["std"]
    return dict(test=test, cols=cols, raw=parse_raw, trace_col=trace_col, trace=parse_trace,
                trees=trees, props=props, center=center, scale=scale)


def load_model(dataset, key, run, cfg):
    width = cfg["raw_len"]
    dim = cfg["feature_dim"]
    vocab = None
    module = None
    if dataset == "Coveo":
        _, classes = load_originals()
        vocab = json.loads((run / "vocabs.json").read_text())
        params = dict(action_vocab_size=5, product_vocab_size=len(vocab[1]),
                      category_vocab_size=len(vocab[2]), max_len=width)
        model = (classes["APCLateFusionTransformer"](**params, z_dim=dim) if dim
                 else classes["APCTransformer"](**params))
    else:
        name = dataset.lower()
        module = original(name)
        model = model_for(name, module, dict(cap=width, z_dim=dim, vocab_sizes=cfg["vocab_sizes"]))
        if dataset == "Cosmetics":
            vocab = json.loads((run / "vocab.json").read_text())
    model.load_state_dict(torch.load(run / "best_model.pt", map_location="cpu", weights_only=True))
    return model.to(DEVICE).eval(), vocab, module


def encode_raw(dataset, source, cfg, vocab, module):
    width = cfg["raw_len"]
    test = source["test"]
    n = len(test)
    dtype = np.int64 if dataset == "Coveo" else np.int32
    keys = ("x",) if dataset == "Cosmetics" else ("action", "product", "category")
    encoded = {key: np.zeros((n, width), dtype=dtype) for key in keys}
    lengths = np.empty(n, dtype=np.int64)
    for i, row in enumerate(test[source["cols"]].itertuples(index=False, name=None)):
        action, product, category = (source["raw"](value)[:width] for value in row)
        length = len(action)
        if not length or len(product) != length or len(category) != length:
            raise ValueError("Invalid raw session fields")
        lengths[i] = length
        if dataset == "Cosmetics":
            tokens = (f"A={a}|P={p}|C={c}" for a, p, c in zip(action, product, category))
            encoded["x"][i, :length] = [vocab.get(token, 1) for token in tokens]
        elif dataset == "Electronics":
            encoded["action"][i, -length:] = [module.ACTION_TO_ID.get(str(value), 0) for value in action]
            encoded["product"][i, -length:] = product
            encoded["category"][i, -length:] = category
        else:
            for key, sequence, dictionary in zip(keys, (action, product, category), vocab):
                encoded[key][i, :length] = [dictionary.get(str(token), 1) for token in sequence]
    if dataset != "Coveo":
        encoded["length"] = lengths
    return encoded


def encode_symbolic(source, condition):
    parse = source["trace"]
    if condition == "counts":
        props = source["props"]
        rows = []
        for value in source["test"][source["trace_col"]]:
            trace = parse(value)
            rows.append([sum(atom in state for state in trace) for atom in props] + [len(trace)])
        return (np.asarray(rows, dtype=np.float32) - source["center"]) / source["scale"]
    limit = 10 if condition == "prefix_ltlf" else None
    return np.stack([vector(parse(value)[:limit], source["trees"])
                     for value in source["test"][source["trace_col"]]]).astype(np.float32)


def benchmark_run(dataset, condition, cfg, source, model, vocab, module, expected, repetitions):
    rows = []
    for repetition in range(repetitions + 1):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        fields = encode_raw(dataset, source, cfg, vocab, module)
        raw_sec = time.perf_counter() - start
        symbolic_start = time.perf_counter()
        if condition in SYMBOLIC:
            fields["z"] = encode_symbolic(source, condition)
        symbolic_sec = time.perf_counter() - symbolic_start
        gpu_start = time.perf_counter()
        fields["y"] = source["test"].purchase.to_numpy(dtype=np.float32)
        dataset_tensors = TensorDataset(*(torch.from_numpy(value) for value in fields.values()))
        collate = trim_full_batch if dataset == "Coveo" and condition == "full_raw" else None
        loader = DataLoader(dataset_tensors, batch_size=cfg.get("eval_batch_size", cfg["batch_size"]),
                            shuffle=False, pin_memory=True, collate_fn=collate)
        if dataset == "Coveo":
            label, probability = coveo_predict(model, loader, DEVICE)
        else:
            label, probability = secondary_predict(model, loader, {"names": list(fields)}, dataset.lower(), DEVICE)
        torch.cuda.synchronize()
        end = time.perf_counter()
        if not np.array_equal(label, expected.label):
            raise ValueError(f"Prediction labels disagree for {dataset}/{condition}")
        abs_error = np.max(np.abs(probability - expected.probability.to_numpy(np.float32)))
        if abs_error > 1e-4:
            raise ValueError(f"Prediction replay discrepancy {abs_error:g} for {dataset}/{condition}")
        if repetition:
            rows.append(dict(dataset=dataset, condition=CONDITIONS[condition], repetition=repetition,
                             n=len(label), raw_encoding_sec=raw_sec, symbolic_feature_sec=symbolic_sec,
                             inference_and_transfers_sec=end - gpu_start, total_sec=end - start,
                             peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                             max_probability_replay_error=abs_error))
    return rows


def plot_tradeoffs(summary, training, prefix):
    fig, axes = plt.subplots(2, summary.dataset.nunique(), figsize=(10.2, 5.2),
                             squeeze=False, layout="constrained")
    colors = ("#555555", "#999999", "#0072B2", "#E69F00", "#009E73")
    for index, (dataset, group) in enumerate(summary.groupby("dataset", sort=False)):
        costs = training[training.dataset == dataset].set_index("condition")
        for row, field, xlabel in (
            (0, "train_val_sec_per_epoch", "Train + validation per epoch (s)"),
            (1, "total_sec_median", "Inference pipeline median (s)"),
        ):
            axis = axes[row, index]
            values = costs[field].loc[group.condition] if row == 0 else group[field]
            for cost, mcc, condition, color in zip(values, group.MCC, group.condition, colors):
                axis.scatter(cost, mcc, color=color, label=condition, s=48, edgecolor="white", linewidth=.4)
            axis.set(xlabel=xlabel, title=dataset if row == 0 else "",
                     ylabel="Test MCC" if index == 0 else "")
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
            axis.grid(alpha=.25)
            axis.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)
    fig.suptitle("Seed-42 accuracy and compute on the local RTX 4060", fontsize=11)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"{prefix}compute_tradeoff.{suffix}", dpi=320)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("Use one or more measured repetitions.")
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the paired runtime benchmark")
    all_rows, training = [], []
    for dataset in (args.dataset,) if args.dataset else DATASETS:
        source = source_for(dataset)
        for condition in CONDITIONS:
            run = ROOT / "runs" / f"{GROUPS[dataset]}_{condition}"
            cfg = json.loads((run / "config.json").read_text())
            metrics = json.loads((run / "metrics.json").read_text())
            expected = pd.read_csv(run / "predictions.csv", dtype={"session_id": str})
            if not np.array_equal(expected.session_id, source["test"].session_id_hash):
                raise ValueError("Test session order mismatch")
            model, vocab, module = load_model(dataset, condition, run, cfg)
            rows = benchmark_run(dataset, condition, cfg, source, model, vocab, module, expected, args.repetitions)
            all_rows.extend(rows)
            training.append(dict(dataset=dataset, condition=CONDITIONS[condition],
                                 feature_precompute_sec=metrics["feature_construction_all_splits_sec"],
                                 training_and_validation_sec=metrics["train_including_validation_sec"],
                                 completed_epochs=metrics["actual_epochs_ran"],
                                 train_val_sec_per_epoch=metrics["train_including_validation_sec"] / metrics["actual_epochs_ran"],
                                 parameter_count=metrics["n_params"], MCC=metrics["mcc"], AUPRC=metrics["auprc"]))
            print(dataset, condition, f"{np.median([row['total_sec'] for row in rows]):.3f}s median", flush=True)
            del model
    measurements = pd.DataFrame(all_rows)
    timing_columns = ["raw_encoding_sec", "symbolic_feature_sec", "inference_and_transfers_sec", "total_sec"]
    grouped = measurements.groupby(["dataset", "condition"], sort=False)
    summary = grouped[timing_columns].median().add_suffix("_median").join(
        grouped.total_sec.agg(["min", "max"]).add_prefix("total_sec_")
    ).join(grouped.peak_cuda_allocated_mib.max()).join(grouped.n.first()).reset_index()
    ref = summary.loc[summary.condition == "Full raw"].set_index("dataset").total_sec_median
    summary["speedup_vs_full_raw"] = summary.dataset.map(ref) / summary.total_sec_median
    summary["median_sessions_per_sec"] = summary.n / summary.total_sec_median
    training = pd.DataFrame(training)
    summary = summary.merge(training[["dataset", "condition", "MCC", "AUPRC"]],
                            on=["dataset", "condition"], validate="one_to_one")
    prefix = f"{args.dataset.lower()}_" if args.dataset else ""
    measurements.to_csv(OUT / f"{prefix}inference_runtime_repetitions.csv", index=False)
    summary.to_csv(OUT / f"{prefix}inference_runtime.csv", index=False)
    training.to_csv(OUT / f"{prefix}training_runtime.csv", index=False)
    tex_table(summary[["dataset", "condition", "n", *[f"{col}_median" for col in timing_columns],
                       "speedup_vs_full_raw"]], OUT / f"{prefix}inference_runtime.tex", ".3f")
    plot_tradeoffs(summary, training, prefix)
    meta = dict(gpu=torch.cuda.get_device_name(), cpu=platform.processor(),
                torch_version=str(torch.__version__), cpu_threads=4,
                repetitions=args.repetitions, warmup_passes=1,
                timing_scope="From loaded session fields to scores, excluding disk I/O, original event abstraction and formula learning.",
                prediction_tolerance=1e-4)
    (OUT / f"{prefix}inference_runtime_protocol.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
