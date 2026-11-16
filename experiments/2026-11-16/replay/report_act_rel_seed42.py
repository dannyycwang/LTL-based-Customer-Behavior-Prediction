"""Build audited comparative tables from saved seed-42 predictions."""
from artifact_paths import (AUDIT_ROOT, WORK_DIR, MODEL_DIR, COVEO_DATA, COVEO_DIAGNOSTIC_DATA, COVEO_BASIS, ELECTRONICS_BASIS, COSMETICS_BASIS, resolve_saved_path)
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_act_rel_seed42 import CONDITIONS, PROTOCOL, ROOT, sha
from report_coveo_seed42 import (
    CONDITIONS as SHORT_LABELS, METRICS, OUT as COVEO, tex_table, write_paired_tables,
)
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, matthews_corrcoef


LABELS = {"full_raw": "Full raw", **SHORT_LABELS}
DATASETS = ("Coveo", "Electronics", "Cosmetics")


def verified_runs(dataset, protocol=PROTOCOL, out=None):
    dataset_key = dataset.lower()
    if out is None:
        out = ROOT / f"{dataset_key}_seed42_comparison"
    out.mkdir(exist_ok=True)
    rows, stratified, predictions = [], [], {}
    reference = None
    reference_config = None
    for condition in CONDITIONS:
        run = ROOT / "runs" / f"{dataset_key}_{protocol}_{condition}"
        if not (run / "metrics.json").is_file():
            continue
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        saved = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        signature = {k: config[k] for k in ("seed", "protocol", "split_source_sha256", "basis_sha256",
                                             "source_sha256", "runner_sha256", "epochs", "patience",
                                             "batch_size", "monitor", "threshold", "raw_window", "vocab_sizes")}
        if reference_config is None:
            reference_config = signature
        if signature != reference_config or config["raw_window"] != "first":
            raise ValueError(f"Inconsistent protocol: {run}")
        if condition != "full_raw" and config["raw_len"] != 10:
            raise ValueError(f"Expected first 10 events: {run}")
        if (saved["actual_epochs_ran"] != config["epochs"]
                and saved["actual_epochs_ran"] - saved["best_epoch"] != config["patience"]):
            raise ValueError(f"Incomplete training: {run}")
        ids = pd.read_csv(run / "test_ids.csv", dtype={"session_id": str}).set_index("session_id")
        pred = pd.read_csv(run / "predictions.csv", dtype={"session_id": str}).set_index("session_id")
        if not ids.index.is_unique or not ids.equals(pred[["label", "length"]]):
            raise ValueError(f"Misaligned prediction labels: {run}")
        if reference is None:
            reference = ids
        if not reference.equals(ids):
            raise ValueError(f"Test IDs differ between conditions: {run}")
        predictions[LABELS[condition]] = pred
        by_length = pd.read_csv(run / "stratified_metrics.csv").set_index("stratum")
        for stratum, mask in (("all", ids.length > 0), ("L<=10", ids.length <= 10), ("L>10", ids.length > 10)):
            subset = pred[mask]
            y = subset.label.to_numpy()
            p = subset.probability.to_numpy(dtype=np.float32)
            binary = p >= config["threshold"]
            if len(np.unique(y)) != 2:
                raise ValueError(f"Unexpected single-class stratum {stratum}: {run}")
            replay = {
                "mcc": matthews_corrcoef(y, binary), "auprc": average_precision_score(y, p),
                "f1": f1_score(y, binary), "brier": brier_score_loss(y, p),
            }
            values = by_length.loc[stratum]
            if len(y) != values["n"] or not np.isclose(y.mean(), values["positive_rate"]):
                raise ValueError(f"Wrong stratum size or rate: {run}/{stratum}")
            if not np.allclose(list(replay.values()), values[list(replay)].to_numpy(), rtol=0, atol=1e-7):
                raise ValueError(f"Metric replay failed: {run}/{stratum}")
            if stratum == "all" and not np.allclose(
                list(replay.values()), [saved[k] for k in replay], rtol=0, atol=1e-7
            ):
                raise ValueError(f"Metric summary replay failed: {run}")
            stratified.append(dict(condition=LABELS[condition], stratum=stratum, n=len(y),
                                   positive_rate=y.mean(), **replay))
        rows.append(dict(condition=LABELS[condition], n=len(ids), raw_cap=config["raw_len"],
                         batch_size=config["batch_size"], **{k: saved[k] for k in (
                             *METRICS, "seed", "threshold", "n_params", "feature_dim", "best_epoch",
                             "actual_epochs_ran", "train_including_validation_sec",
                             "feature_construction_all_splits_sec", "inference_precomputed_features_sec", "gpu",
                         )}, prediction_sha256=sha(run / "predictions.csv")))

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result.to_csv(out / "run_summary.csv", index=False)
    tex_table(result[["condition", *METRICS, "threshold", "feature_dim", "n_params"]],
              out / "predictive_metrics.tex", ".4f")
    length_results = pd.DataFrame(stratified)
    length_results.to_csv(out / "length_strata.csv", index=False)
    tex_table(length_results, out / "length_strata.tex", ".4f")
    write_paired_tables(length_results, predictions, out)
    runtime = result[["condition", "feature_construction_all_splits_sec", "train_including_validation_sec",
                      "inference_precomputed_features_sec"]].copy()
    runtime.columns = ["Condition", "Feature (s)", "Train + validation (s)", "Test inference (s)"]
    tex_table(runtime, out / "runtime.tex", ".2f")
    return result


def plot_summary(rows, path):
    colors = ["#666666", "#949494", "#0072B2", "#E69F00", "#009E73"]
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(8.2, 8.1), layout="constrained")
    for index, dataset in enumerate(DATASETS):
        subset = rows[rows.dataset == dataset].set_index("condition").loc[list(LABELS.values())]
        for axis, metric in zip(axes[index], ("MCC", "AUPRC")):
            for offset, (color, score) in enumerate(zip(colors, subset[metric])):
                if np.isfinite(score):
                    axis.barh(offset, score, color=color, height=0.68)
                    axis.text(score + 0.02, offset, f"{score:.3f}", va="center", fontsize=8)
                else:
                    axis.text(0.015, offset, "Pending", va="center", fontsize=8, color="#555555")
            axis.set(yticks=range(len(subset)), yticklabels=subset.index if metric == "MCC" else [],
                     ylim=(len(subset) - 0.5, -0.5), xlim=(0, 1.09), title=f"{dataset} · {metric}")
            axis.spines[["top", "right", "left"]].set_visible(False)
            axis.grid(axis="x", alpha=0.25)
            axis.set_axisbelow(True)
    fig.savefig(path / "predictive_metrics.pdf")
    fig.savefig(path / "predictive_metrics.png", dpi=320)
    plt.close(fig)


def formula_scope_summary(out):
    rows = []
    for dataset in DATASETS:
        directory = ROOT / f"{dataset.lower()}_seed42_comparison"
        rates = pd.read_csv(directory / "formula_changes.csv")
        scope = rates[rates.formula_id == "ANY_BIT"].set_index("stratum")
        shuffles = pd.read_csv(directory / "shuffle_changes.csv")
        shuffle_rates = shuffles.loc[shuffles.formula_id == "ANY_BIT", "changed_rate"]
        rows.append(dict(dataset=dataset, n_test=int(scope.loc["all", "n"]),
                         n_long=int(scope.loc["L>10", "n"]),
                         long_fraction=scope.loc["L>10", "n"] / scope.loc["all", "n"],
                         long_vector_changed_fraction=scope.loc["L>10", "changed_rate"],
                         all_vector_changed_fraction=scope.loc["all", "changed_rate"],
                         shuffled_vector_changed_mean=shuffle_rates.mean(),
                         shuffled_vector_changed_sd=shuffle_rates.std()))
    pd.DataFrame(rows).to_csv(out / "formula_scope.csv", index=False)


def main():
    summaries = [pd.read_csv(COVEO / "run_summary.csv").assign(dataset="Coveo")]
    for dataset in DATASETS[1:]:
        frame = verified_runs(dataset)
        if not frame.empty:
            summaries.append(frame.assign(dataset=dataset))
    results = pd.concat(summaries, ignore_index=True)
    target = pd.MultiIndex.from_product([DATASETS, LABELS.values()], names=["dataset", "condition"])
    results = results.set_index(["dataset", "condition"]).reindex(target).reset_index()
    results = results.rename(columns={"mcc": "MCC", "auprc": "AUPRC", "f1": "F1", "brier": "Brier"})
    columns = ["dataset", "condition", "MCC", "AUPRC", "F1", "Brier"]
    out = ROOT / "act_rel_seed42_summary"
    out.mkdir(exist_ok=True)
    results.to_csv(out / "results.csv", index=False)
    tex_table(results[columns], out / "predictive_metrics.tex", ".4f")
    plot_summary(results, out)
    formula_scope_summary(out)
    for name in ("paired_deltas", "length_strata"):
        tables = []
        for dataset in DATASETS:
            path = ROOT / f"{dataset.lower()}_seed42_comparison" / f"{name}.csv"
            if path.is_file():
                table = pd.read_csv(path)
                table.insert(0, "dataset", dataset)
                tables.append(table)
        if tables:
            table = pd.concat(tables, ignore_index=True)
            table.to_csv(out / f"{name}.csv", index=False)
            if name == "paired_deltas":
                export = table[["dataset", "contrast", "stratum", "n", "delta_mcc",
                                "delta_auprc", "delta_f1", "delta_brier"]]
            else:
                export = table[["dataset", "condition", "stratum", "n", "positive_rate", *METRICS]]
            tex_table(export, out / f"{name}.tex", ".4f")
    status = results[columns[:4]].copy()
    status["status"] = np.where(status.MCC.notna(), "Completed seed 42", "Downstream training pending")
    status.to_csv(ROOT / "new_experiment_status.csv", index=False)
    print(results[columns].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
