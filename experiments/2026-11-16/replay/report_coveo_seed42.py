"""Generate the seed-42 comparison from saved protected-protocol runs."""
from artifact_paths import (AUDIT_ROOT, WORK_DIR, MODEL_DIR, COVEO_DATA, COVEO_DIAGNOSTIC_DATA, COVEO_BASIS, ELECTRONICS_BASIS, COSMETICS_BASIS, resolve_saved_path)
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diagnose_coveo import sha


ROOT = AUDIT_ROOT
LOCAL_PACKAGES = WORK_DIR / "python_packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, matthews_corrcoef

OUT = ROOT / "coveo_seed42_comparison"
CONDITIONS = {
    "full_raw": "Full raw",
    "short_raw": "Short raw",
    "prefix_ltlf": "Short+Prefix-LTLf",
    "counts": "Short+Counts",
    "full_ltlf": "Short+Full-LTLf",
}
METRICS = ["mcc", "auprc", "f1", "brier"]


def tex_table(frame, path, fmt):
    def escape(value):
        return str(value).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

    def cell(value):
        if pd.isna(value):
            return "NA"
        return format(value, fmt) if isinstance(value, (float, np.floating)) else escape(value)

    lines = [
        r"\begin{tabular}{" + "l" + "r" * (len(frame.columns) - 1) + "}",
        r"\toprule",
        " & ".join(escape(c) for c in frame.columns) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(cell(v) for v in row) + r" \\" for row in frame.itertuples(index=False, name=None))
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_results():
    reference_config = None
    short_runner = None
    reference_labels = None
    models = []
    strata = []
    predictions = {}
    for key, label in CONDITIONS.items():
        run = ROOT / "runs" / f"protected_first_v1_seed42_{key}"
        if not (run / "metrics.json").is_file():
            continue
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        matches = ["dataset_sha256", "basis_sha256", "model_source_sha256",
                   "evaluator_source_sha256", "splits", "raw_window",
                   "batch_size", "epochs", "patience", "threshold_protocol"]
        signature = {name: config[name] for name in matches}
        if reference_config is None:
            reference_config = signature
        if signature != reference_config or config["raw_window"] != "first":
            raise ValueError(f"Protocol mismatch for {key}")
        if key == "full_raw":
            if config["truncated_sessions"] != 0 or config["feature_dim"] != 0:
                raise ValueError("Full raw must use the whole sequence without symbolic features")
        else:
            if short_runner is None:
                short_runner = config["runner_sha256"]
            if config["raw_len"] != 10 or config["runner_sha256"] != short_runner:
                raise ValueError(f"Short-run mismatch for {key}")
        for split, digest in config["splits"].items():
            if sha(run / f"{split}_ids.csv") != digest:
                raise ValueError(f"Split hash mismatch for {key}/{split}")
        pred = pd.read_csv(run / "predictions.csv", dtype={"session_id": str}).set_index("session_id")
        test_ids = pd.read_csv(run / "test_ids.csv", dtype={"session_id": str}).set_index("session_id")
        if not pred.index.is_unique or not pred[["label"]].equals(test_ids[["label"]]):
            raise ValueError(f"Prediction IDs/labels differ from test CSV for {key}")
        if key == "full_raw" and pred.length.max() > config["raw_len"]:
            raise ValueError("Full raw width does not cover the test sequences")
        labels = pred[["label", "length"]]
        if reference_labels is None:
            reference_labels = labels
        if not labels.equals(reference_labels):
            raise ValueError(f"Pairing mismatch for {key}")
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        if metrics["actual_epochs_ran"] != config["epochs"]:
            raise ValueError(f"Incomplete training for {key}")
        model = {name: metrics[name] for name in METRICS}
        model.update({
            "condition": label, "seed": 42, "n": len(pred), "threshold": metrics["threshold"],
            "raw_cap": config["raw_len"],
            "feature_dim": metrics["feature_dim"], "parameters": metrics["n_params"],
            "epochs": metrics["actual_epochs_ran"], "best_epoch": metrics["best_epoch"],
            "batch_size": config["batch_size"], "gpu": metrics["gpu"],
            "feature_construction_all_splits_sec": metrics["feature_construction_all_splits_sec"],
            "train_including_validation_sec": metrics["train_including_validation_sec"],
            "test_inference_precomputed_sec": metrics["inference_precomputed_features_sec"],
            "metrics_sha256": sha(run / "metrics.json"),
            "predictions_sha256": sha(run / "predictions.csv"),
        })
        models.append(model)
        frame = pd.read_csv(run / "stratified_metrics.csv")
        for stratum, subset in (
            ("all", pred), ("L<=10", pred[pred.length <= 10]), ("L>10", pred[pred.length > 10])
        ):
            saved = frame.set_index("stratum").loc[stratum]
            if int(saved["n"]) != len(subset) or not np.isclose(saved["positive_rate"], subset.label.mean()):
                raise ValueError(f"Stratum counts/labels differ for {key}/{stratum}")
            probability = subset.probability.to_numpy(dtype=np.float32)
            decision = probability >= metrics["threshold"]
            recomputed = [
                matthews_corrcoef(subset.label, decision),
                average_precision_score(subset.label, probability),
                f1_score(subset.label, decision),
                brier_score_loss(subset.label, probability),
            ]
            if not np.allclose(saved[METRICS].to_numpy(dtype=float), recomputed, atol=1e-7, rtol=0):
                raise ValueError(f"Metric replay differs for {key}/{stratum}")
            if stratum == "all" and not np.allclose(
                [metrics[name] for name in METRICS], recomputed, atol=1e-7, rtol=0
            ):
                raise ValueError(f"Summary metric replay differs for {key}")
        frame.insert(0, "condition", label)
        strata.append(frame)
        predictions[label] = pred
    return pd.DataFrame(models), strata, predictions


def write_paired_tables(strata, predictions, directory=OUT):
    full_name = CONDITIONS["full_ltlf"]
    if full_name not in predictions:
        return
    full = predictions[full_name]
    rows = []
    session_pairs = full[["label", "length", "probability"]].rename(
        columns={"probability": "full_ltlf_probability"}
    )
    for key in ("prefix_ltlf", "counts"):
        name = CONDITIONS[key]
        if name not in predictions:
            continue
        other = predictions[name]
        delta = full.probability - other.probability
        session_pairs[f"{key}_probability"] = other.probability
        session_pairs[f"full_minus_{key}_probability"] = delta
        for group, values in strata.groupby("stratum", sort=False):
            by_condition = values.set_index("condition")
            if group == "all":
                subset = delta
            elif group == "L<=10":
                subset = delta[full.length <= 10]
            else:
                subset = delta[full.length > 10]
            difference = by_condition.loc[full_name, METRICS] - by_condition.loc[name, METRICS]
            rows.append({
                "contrast": f"Full-LTLf minus {name.split('+', 1)[1]}",
                "stratum": group, "n": len(subset),
                **{f"delta_{metric}": difference[metric] for metric in METRICS},
                "mean_probability_delta": subset.mean(),
                "mean_absolute_probability_delta": subset.abs().mean(),
            })
    paired = pd.DataFrame(rows)
    if paired.empty:
        return
    paired.to_csv(directory / "paired_deltas.csv", index=False)
    session_pairs.to_csv(directory / "paired_predictions.csv.gz")
    tex_table(paired.iloc[:, :7], directory / "paired_deltas.tex", ".4f")


def draw_metrics(models):
    colors = ["#666666", "#949494", "#0072B2", "#E69F00", "#009E73"]
    order = list(CONDITIONS.values())
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.1), layout="constrained")
    for axis, metric, title in zip(axes, ("mcc", "auprc"), ("MCC", "AUPRC (average precision)")):
        for i, row in enumerate(models.itertuples(index=False)):
            score = getattr(row, metric)
            axis.barh(i, score, color=colors[order.index(row.condition)], height=0.62)
            axis.text(score + 0.016, i, f"{score:.3f}", va="center", fontsize=9)
        axis.set(yticks=range(len(models)), yticklabels=models.condition if metric == "mcc" else [],
                 title=title, xlim=(0, 1.04))
        axis.invert_yaxis()
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)
    fig.suptitle("Coveo · seed 42 · 38,490 test sessions", fontsize=11)
    fig.savefig(OUT / "predictive_metrics.pdf")
    fig.savefig(OUT / "predictive_metrics.png", dpi=320)
    plt.close(fig)


def draw_formula_diagnostics(directory=OUT):
    path = directory / "formula_changes.csv"
    if not path.is_file():
        return
    changes = pd.read_csv(path)
    changed_long = changes[(changes.stratum == "L>10") & (changes.formula_id != "ANY_BIT")]
    shuffle = pd.read_csv(directory / "shuffle_changes.csv")
    shuffled = shuffle.groupby("formula_id").changed_rate.agg(["mean", "std"]).reindex(changed_long.formula_id)
    x = np.arange(len(changed_long))
    forward = changed_long.zero_to_one_rate * 100
    backward = changed_long.one_to_zero_rate * 100
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1), layout="constrained")
    axes[0].bar(x, forward, color="#0072B2", label="0 → 1")
    axes[0].bar(x, backward, bottom=forward, color="#D55E00", label="1 → 0")
    axes[0].set_ylim(0, max(forward + backward) * 1.22)
    axes[0].legend(frameon=False, loc="upper left", ncols=2)
    axes[0].set(title=f"Prefix → full, L > 10 (n={int(changed_long.n.iloc[0]):,})")
    axes[1].bar(x, shuffled["mean"] * 100, yerr=shuffled["std"] * 100,
                color="#009E73", capsize=2)
    axes[1].set(title="Order shuffle, all test sessions")
    axes[1].set_xlabel("Mean ± SD across 10 shuffles")
    for axis in axes:
        axis.set(xticks=x, xticklabels=changed_long.formula_id, ylabel="Changed satisfaction (%)")
        axis.tick_params(axis="x", labelsize=8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    fig.savefig(directory / "formula_scope_and_shuffle.pdf")
    fig.savefig(directory / "formula_scope_and_shuffle.png", dpi=320)
    plt.close(fig)

    case_path = directory / "real_trace_case.json"
    if not case_path.is_file():
        return
    case = json.loads(case_path.read_text(encoding="utf-8"))
    trace = case["trace"]
    atoms = sorted(set().union(*map(set, trace)))
    presence = [[int(atom in state) for state in trace] for atom in atoms]
    from matplotlib.colors import ListedColormap
    fig, axis = plt.subplots(figsize=(8.2, 2.7), layout="constrained")
    axis.imshow(presence, cmap=ListedColormap(["#F4F4F4", "#0072B2"]),
                vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    axis.axvline(9.5, color="#D55E00", linewidth=2)
    axis.set(xticks=range(len(trace)), xticklabels=range(1, len(trace) + 1),
             yticks=range(len(atoms)), yticklabels=atoms, xlabel="Event position",
             ylabel="Proposition",
             title=f"{case['formula_id']}  {case['formula']}\n"
                   f"First 10: false · Full trace: true · First true at event {case['first_true_length']}")
    fig.savefig(directory / "real_trace_case.pdf")
    fig.savefig(directory / "real_trace_case.png", dpi=320)
    plt.close(fig)


def write_notes(models):
    config = json.loads((ROOT / "runs" / "protected_first_v1_seed42_short_raw" / "config.json").read_text())
    text = f"""Coveo Act+Rel, seed 42

The comparison uses the protected seed-42 split. The 20,000 sessions used to fit
and select the saved formula basis are in the training split. Each short encoder
receives action, product ID and category ID from the first 10 available events.
Prefix-LTLf and Full-LTLf use the same 10 formulas. Counts uses 13 proposition
counts plus trace length from the same complete symbolic trace and is standardized
with training statistics.

Each condition has one run. The checkpoint is selected by validation
F1 at threshold 0.5, and the test decision threshold is selected by validation F1
on the original 0.05 to 0.95 grid. AUPRC is average precision. Length groups use
the same threshold as the overall evaluation for their respective model.

The runtime table reports seconds. Feature time covers all splits and includes
trace parsing for the fusion conditions. Training includes validation and
checkpoint writes. Test inference uses precomputed features. Formula learning
was performed previously and its cost is outside these timings. Other audit work
ran concurrently, so the timings describe the observed runs.

GPU: {models.gpu.iloc[0]}
Batch size: {config['batch_size']}
Planned epochs per condition: {config['epochs']}
Test IDs SHA-256: {config['splits']['test']}
Formula basis SHA-256: {config['basis_sha256']}

The paired tables align by session ID, label and length. Differences are computed
on the same 38,490 test sessions. Prefix and full satisfaction vectors agree
exactly on all 27,349 sessions of length at most 10. Predictors are fitted
separately across the full training set, so their predictions may differ on
those sessions. The shuffle figure displays diagnostic sensitivity with
standard deviations across the 10 symbolic shuffles.

In the illustrated test session, event 11 has both add-to-cart (a) and product
revisit (rp). It provides the two-step look-ahead for X(X(a)) at event 9 and
the three-step look-ahead for X(X(X(rp))) at event 8, which satisfies
true U X(X(X(rp))) at every position before event 9. The case was chosen by
taking the first basis formula containing rp with a false-to-true change,
then the shortest matching test session with ties broken by session ID.

The predictive comparison is descriptive for seed 42.
"""
    if "Full raw" in set(models.condition):
        max_length = int(models.loc[models.condition == "Full raw", "raw_cap"].iloc[0])
        text += f"""
Full raw uses action, product ID and category ID for all observed events, with
a maximum session length of {max_length}. Batch padding skips columns common
to no real event and preserves the original positional IDs. The same model,
test IDs, training batch size and validation procedure are used as in Short raw.
"""
    (OUT / "run_notes.txt").write_text(text, encoding="utf-8")


def main():
    models, strata, predictions = read_results()
    if models.empty:
        print("No completed runs.")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    score_cols = ["condition", "mcc", "auprc", "f1", "brier", "threshold", "feature_dim", "parameters"]
    models.to_csv(OUT / "run_summary.csv", index=False)
    tex_table(models[score_cols], OUT / "predictive_metrics.tex", ".4f")
    strata = pd.concat(strata, ignore_index=True)
    strata.to_csv(OUT / "length_strata.csv", index=False)
    tex_table(strata[["condition", "stratum", "n", "positive_rate", *METRICS]],
              OUT / "length_strata.tex", ".4f")
    runtime = models[["condition", "feature_construction_all_splits_sec",
                      "train_including_validation_sec", "test_inference_precomputed_sec"]]
    runtime = runtime.rename(columns={
        "condition": "Condition", "feature_construction_all_splits_sec": "Feature (s)",
        "train_including_validation_sec": "Train + validation (s)",
        "test_inference_precomputed_sec": "Test inference (s)",
    })
    tex_table(runtime, OUT / "runtime.tex", ".2f")
    write_paired_tables(strata, predictions)
    draw_metrics(models)
    draw_formula_diagnostics()
    write_notes(models)
    print(models[score_cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
