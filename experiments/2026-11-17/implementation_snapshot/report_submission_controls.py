"""Verify initialization replicates and plot the new matched CPU controls."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bootstrap_submission_controls import GROUPS, cluster_ids, paired_interval
from report_coveo_seed42 import tex_table
from run_submission_controls import DATASETS, OUT, ROOT
from sklearn.metrics import average_precision_score, matthews_corrcoef


CPU_CONTRASTS = {
    "full_observation": [
        ("Short + action counts", "Short + length"),
        ("Short + Act+Rel counts", "Short + action counts"),
        ("Short + Full-LTLf", "Short + Act+Rel counts and length"),
    ],
    "fixed_event_horizon": [
        ("Short + Prefix-LTLf", "Short raw"),
        ("Short + Prefix-LTLf", "Short + Act+Rel counts"),
    ],
}


def control_intervals(scores):
    rng = np.random.default_rng(20261113)
    replicates = 1500
    rows = []
    for (dataset, protocol), metrics in scores.groupby(["dataset", "protocol"], sort=False):
        path = OUT / f"{dataset.lower()}_{protocol}_predictions.csv.gz"
        pred = pd.read_csv(path, dtype={"session_id": str}).set_index("session_id", verify_integrity=True)
        groups, unit = cluster_ids(dataset, pred.index)
        metrics = metrics.set_index("condition")
        for a, b in CPU_CONTRASTS[protocol]:
            point, lower, upper = paired_interval(pred.label.to_numpy(np.int64),
                pred[a].to_numpy(), pred[b].to_numpy(), metrics.loc[a, "threshold"],
                metrics.loc[b, "threshold"], groups, replicates, rng)
            expected = metrics.loc[a, ["MCC", "AUPRC"]].to_numpy(float) - metrics.loc[b, ["MCC", "AUPRC"]].to_numpy(float)
            if not np.allclose(point, expected, atol=1e-7, rtol=0):
                raise ValueError(f"CPU metric replay differs: {dataset}/{protocol}/{a}")
            for index, metric in enumerate(("MCC", "AUPRC")):
                rows.append(dict(dataset=dataset, protocol=protocol, first=a, second=b, n_sessions=len(pred),
                                 unit=unit, replicates=replicates, metric=metric, delta=point[index],
                                 lower_95=lower[index], upper_95=upper[index]))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "cpu_paired_bootstrap.csv", index=False)


def plot_controls(scores):
    colors = {"Coveo": "#0072B2", "Electronics": "#009E73", "Cosmetics": "#D55E00"}
    for protocol, frame in scores.groupby("protocol", sort=False):
        names = frame.condition.drop_duplicates().tolist()
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 1.8 + len(names) * .47), layout="constrained")
        for axis, metric in zip(axes, ("MCC", "AUPRC")):
            for offset, dataset in enumerate(DATASETS):
                group = frame[frame.dataset == dataset].set_index("condition").loc[names]
                axis.barh(np.arange(len(names)) + (offset - 1) * .24, group[metric],
                          height=.22, label=dataset, color=colors[dataset])
            axis.set(yticks=np.arange(len(names)), yticklabels=names if metric == "MCC" else [],
                     xlim=(0, 1), title=metric)
            axis.invert_yaxis()
            axis.spines[["top", "right", "left"]].set_visible(False)
            axis.grid(axis="x", alpha=.23)
            axis.set_axisbelow(True)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)
        title = "Full-observation controls" if protocol == "full_observation" else "Predicting purchase in events 11–20 from the first 10"
        fig.suptitle(f"{title} · logistic models · seed-42 split", fontsize=11)
        for suffix in ("pdf", "png"):
            fig.savefig(OUT / f"{protocol}.{suffix}", dpi=320)
        plt.close(fig)


def initialization_results():
    rows = []
    for dataset, prefix in GROUPS.items():
        for condition in ("counts", "full_ltlf"):
            root = ROOT / "runs"
            baseline = root / f"{prefix}_{condition}"
            base_cfg = json.loads((baseline / "config.json").read_text())
            base_labels = pd.read_csv(baseline / "predictions.csv", dtype={"session_id": str})
            cols = ["session_id", "label", "length"]
            for init_seed in (42, 43, 44):
                run = baseline if init_seed == 42 else root / f"{prefix}_{condition}_init{init_seed}"
                if not (run / "metrics.json").is_file():
                    continue
                config = json.loads((run / "config.json").read_text())
                compare = ("seed", "basis_sha256", "condition", "epochs", "patience", "batch_size", "raw_len")
                compare += ("dataset_sha256", "splits", "model_source_sha256") if dataset == "Coveo" else (
                    "split_source_sha256", "source_sha256", "monitor", "threshold", "feature_sha256",
                )
                if any(config[key] != base_cfg[key] for key in compare) or config.get("init_seed", 42) != init_seed:
                    raise ValueError(f"Initialization config mismatch: {run}")
                result = json.loads((run / "metrics.json").read_text())
                pred = pd.read_csv(run / "predictions.csv", dtype={"session_id": str})
                if not pred[cols].equals(base_labels[cols]) or not pred.session_id.is_unique:
                    raise ValueError(f"Test IDs differ: {run}")
                prob = pred.probability.to_numpy(np.float32)
                mcc = matthews_corrcoef(pred.label, prob >= result["threshold"])
                ap = average_precision_score(pred.label, prob)
                if not np.allclose([mcc, ap], [result["mcc"], result["auprc"]], atol=1e-7, rtol=0):
                    raise ValueError(f"Metric replay differs: {run}")
                rows.append(dict(dataset=dataset, split_seed=42, downstream_seed=init_seed,
                                 condition=condition, n_test=len(pred), MCC=mcc, AUPRC=ap,
                                 best_epoch=result["best_epoch"], epochs=result["actual_epochs_ran"],
                                 feature_precompute_sec=result["feature_construction_all_splits_sec"],
                                 train_validation_sec=result["train_including_validation_sec"],
                                 inference_precomputed_sec=result["inference_precomputed_features_sec"],
                                 train_val_sec_per_epoch=result["train_including_validation_sec"] / result["actual_epochs_ran"]))
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "neural_initializations.csv", index=False)
    contrasts = []
    for (dataset, seed), group in runs.groupby(["dataset", "downstream_seed"]):
        if len(group) == 2:
            scores = group.set_index("condition")
            values = scores.loc["full_ltlf", ["MCC", "AUPRC"]] - scores.loc["counts", ["MCC", "AUPRC"]]
            contrasts.append(dict(dataset=dataset, downstream_seed=seed,
                                  full_minus_counts_mcc=values["MCC"], full_minus_counts_auprc=values["AUPRC"]))
    paired = pd.DataFrame(contrasts)
    paired.to_csv(OUT / "neural_initialization_deltas.csv", index=False)
    if len(paired) > len(GROUPS):
        summary = paired.groupby("dataset")[["full_minus_counts_mcc", "full_minus_counts_auprc"]].agg(
            ["count", "mean", "std", "min", "max"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary.to_csv(OUT / "neural_initialization_summary.csv")
        tex_table(paired, OUT / "neural_initialization_deltas.tex", ".4f")
    print(paired.to_string(index=False), flush=True)


def main():
    scores = pd.read_csv(OUT / "metrics.csv")
    plot_controls(scores)
    control_intervals(scores)
    initialization_results()


if __name__ == "__main__":
    main()
