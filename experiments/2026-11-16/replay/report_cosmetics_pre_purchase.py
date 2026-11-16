"""Compare the two Cosmetics observation protocols on the common test cohort."""
from artifact_paths import (AUDIT_ROOT, WORK_DIR, MODEL_DIR, COVEO_DATA, COVEO_DIAGNOSTIC_DATA, COVEO_BASIS, ELECTRONICS_BASIS, COSMETICS_BASIS, resolve_saved_path)
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from report_act_rel_seed42 import CONDITIONS, LABELS, METRICS, PROTOCOL, verified_runs
from report_coveo_seed42 import tex_table
from run_act_rel_seed42 import PRE_PURCHASE_OUT, PRE_PURCHASE_PROTOCOL, ROOT
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, matthews_corrcoef


def predict_file(protocol, condition):
    path = ROOT / "runs" / f"cosmetics_{protocol}_{condition}" / "predictions.csv"
    pred = pd.read_csv(path, dtype={"session_id": str}).set_index("session_id", verify_integrity=True)
    if set(pred.threshold) != {0.5}:
        raise ValueError(f"Unexpected threshold: {path}")
    return pred


def metrics(frame):
    y = frame.label.to_numpy()
    p = frame.probability.to_numpy(dtype=np.float32)
    decision = p >= 0.5
    return dict(mcc=matthews_corrcoef(y, decision), auprc=average_precision_score(y, p),
                f1=f1_score(y, decision), brier=brier_score_loss(y, p))


def plot_comparison(frame):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), layout="constrained")
    for axis, name in zip(axes, ("mcc", "auprc")):
        row_pos = np.arange(len(frame))
        axis.barh(row_pos - .17, frame[f"historical_{name}"], height=.30,
                  color="#0072B2", label="Purchase-masked")
        axis.barh(row_pos + .17, frame[f"pre_purchase_{name}"], height=.30,
                  color="#E69F00", label="First-purchase cutoff")
        axis.set(yticks=row_pos, yticklabels=frame.condition if name == "mcc" else [],
                 xlim=(0, 1), title=name.upper())
        axis.invert_yaxis()
        axis.set_axisbelow(True)
        axis.grid(axis="x", alpha=.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False)
    fig.suptitle(f"Cosmetics · seed 42 · {int(frame.n.iloc[0]):,} shared test sessions")
    for suffix in ("pdf", "png"):
        fig.savefig(PRE_PURCHASE_OUT / f"observation_protocol_comparison.{suffix}", dpi=320)
    plt.close(fig)


def write_three_dataset_cutoff_table(runs):
    out = ROOT / "act_rel_seed42_summary"
    primary = pd.read_csv(out / "results.csv")
    columns = ["dataset", "condition", "n", "MCC", "AUPRC", "F1", "Brier"]
    cut = runs.assign(dataset="Cosmetics").rename(columns={
        "mcc": "MCC", "auprc": "AUPRC", "f1": "F1", "brier": "Brier",
    })
    result = pd.concat([primary.loc[primary.dataset != "Cosmetics", columns], cut[columns]],
                       ignore_index=True)
    if len(result) != 15 or result[columns[3:]].isna().any().any():
        raise ValueError("Expected all 15 first-purchase-cutoff results")
    result["n"] = result.n.astype(int)
    result.to_csv(out / "first_purchase_results.csv", index=False)
    tex_table(result, out / "first_purchase_results.tex", ".4f")


def main():
    out = PRE_PURCHASE_OUT
    runs = verified_runs("Cosmetics", protocol=PRE_PURCHASE_PROTOCOL, out=out)
    if len(runs) != len(CONDITIONS):
        print(f"{len(runs)}/{len(CONDITIONS)} pre-purchase runs verified.", flush=True)
        return
    history = pd.read_csv(out / "observation_changes.csv.gz", dtype={"session_id_hash": str})
    eligible = history.loc[(history.split == "test") & history.eligible].set_index("session_id_hash")
    rows = []
    for condition in CONDITIONS:
        old = predict_file(PROTOCOL, condition)
        new = predict_file(PRE_PURCHASE_PROTOCOL, condition)
        if not set(new.index) == set(eligible.index) or not set(new.index) <= set(old.index):
            raise ValueError(f"Common test cohort mismatch: {condition}")
        cohort = eligible.loc[new.index]
        old = old.loc[new.index]
        if not (old.label.equals(new.label) and np.array_equal(cohort.purchase, new.label)
                and np.array_equal(cohort.n_events, old.length)
                and np.array_equal(cohort.pre_purchase_n, new.length)):
            raise ValueError(f"Test labels or lengths differ from observation audit: {condition}")
        historical_metrics, prospective_metrics = metrics(old), metrics(new)
        rows.append(dict(
            condition=LABELS[condition], n=len(new), positive_rate=new.label.mean(),
            **{f"historical_{key}": historical_metrics[key] for key in METRICS},
            **{f"pre_purchase_{key}": prospective_metrics[key] for key in METRICS},
            **{f"delta_{key}": prospective_metrics[key] - historical_metrics[key] for key in METRICS},
        ))
    result = pd.DataFrame(rows)
    result.to_csv(out / "observation_protocol_comparison.csv", index=False)
    columns = ["condition", "n", "historical_mcc", "pre_purchase_mcc", "delta_mcc",
               "historical_auprc", "pre_purchase_auprc", "delta_auprc"]
    tex_table(result[columns], out / "observation_protocol_comparison.tex", ".4f")
    plot_comparison(result)
    write_three_dataset_cutoff_table(runs)
    audit = json.loads((out / "observation_boundary.json").read_text(encoding="utf-8"))
    print(result[columns].to_string(index=False), flush=True)
    print(f"Verified {len(runs)} runs on {audit['splits'][-1]['pre_purchase_n']:,} test sessions.", flush=True)


if __name__ == "__main__":
    main()
