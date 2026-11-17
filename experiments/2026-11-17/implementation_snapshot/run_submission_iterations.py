"""Test conditional logic value and robustness to the future event horizon."""
import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bootstrap_submission_controls import cluster_ids, paired_interval
from report_coveo_seed42 import tex_table
from run_submission_controls import (
    DATASETS, HORIZON, ROOT, SPLITS, audited_inputs, counts_and_prefix, fit_controls, verify_test_replay,
)


OUT = ROOT / "submission_iterations_seed42"
COUNT = "Short + Counts"
LOGIC = "Short + Counts + LTLf"
TRANS = "Short + Counts + transitions"
BOTH = "Short + Counts + transitions + LTLf"
CONTRASTS = ((LOGIC, COUNT), (TRANS, COUNT), (LOGIC, TRANS), (BOTH, TRANS))
HORIZONS = (5, 10, 20)


def task_data(source, future_events=None):
    if future_events is None:
        return source
    output = {}
    for split, table in source.items():
        eligible = table["length"] >= HORIZON
        new = {key: value[eligible] for key, value in table.items()}
        new["label"] = ((new["label"] == 1) & (new["length"] < HORIZON + future_events)).astype(int)
        if np.unique(new["label"]).size != 2:
            raise ValueError(f"Both classes required: {split} / horizon {future_events}")
        output[split] = new
    return output


def transition_selection(table, key, props):
    bits = table[key]
    y = table["label"] == 1
    positive_rate, negative_rate = bits[y].mean(axis=0), bits[~y].mean(axis=0)
    difference = positive_rate - negative_rate
    support = bits.sum(axis=0)
    candidates = np.flatnonzero((support >= 20) & (support <= len(bits) - 20))
    ranked = sorted(candidates, key=lambda i: (-abs(difference[i]), int(i)))[:10]
    if len(ranked) != 10:
        raise ValueError("Expected ten nonconstant transition features")
    rows = []
    for index, i in enumerate(ranked, 1):
        left, right = divmod(i, len(props))
        rows.append(dict(rank=index, from_atom=props[left], to_atom=props[right],
                         train_positive_rate=positive_rate[i], train_negative_rate=negative_rate[i],
                         train_n_present=int(support[i])))
    return ranked, rows


def feature_suite(tables, selected, future):
    scope = "full" if future is None else "prefix"
    key = f"transition_{scope}"
    numerical = {}
    for split, data in tables.items():
        if future is None:
            counts = np.column_stack([data["counts"], data["length"]])
        else:
            counts = data["prefix_counts"]
        logic = data[f"{scope}_formula"]
        ordered = data[key][:, selected]
        numerical[split] = {
            "Short raw": None,
            COUNT: counts,
            LOGIC: np.column_stack([counts, logic]),
            TRANS: np.column_stack([counts, ordered]),
            BOTH: np.column_stack([counts, ordered, logic]),
        }
    return numerical


def uncertainty(scores, replicates=1500):
    rng = np.random.default_rng(201142)
    rows = []
    for (dataset, protocol), metrics in scores.groupby(["dataset", "protocol"], sort=False):
        pred = pd.read_csv(OUT / f"{dataset.lower()}_{protocol}_predictions.csv.gz",
                           dtype={"session_id": str}).set_index("session_id", verify_integrity=True)
        groups, unit = cluster_ids(dataset, pred.index)
        labels = pred.label.to_numpy(np.int64)
        metrics = metrics.set_index("condition")
        for first, second in CONTRASTS:
            point, lower, upper = paired_interval(labels, pred[first].to_numpy(), pred[second].to_numpy(),
                metrics.loc[first, "threshold"], metrics.loc[second, "threshold"], groups, replicates, rng)
            expected = metrics.loc[first, ["MCC", "AUPRC"]].to_numpy(float) - metrics.loc[second, ["MCC", "AUPRC"]].to_numpy(float)
            if not np.allclose(point, expected, atol=1e-7, rtol=0):
                raise ValueError(f"Unexpected contrast replay: {dataset}/{protocol}/{first}")
            for i, metric in enumerate(("MCC", "AUPRC")):
                rows.append(dict(dataset=dataset, protocol=protocol, first=first, second=second,
                                 n_sessions=len(pred), unit=unit, metric=metric, delta=point[i],
                                 lower_95=lower[i], upper_95=upper[i]))
    return pd.DataFrame(rows)


def plot_horizons(scores):
    colors = {"Short raw": "#666666", COUNT: "#E69F00", LOGIC: "#009E73",
              TRANS: "#0072B2", BOTH: "#D55E00"}
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 5.7), layout="constrained")
    horizons = [f"next_{future}" for future in HORIZONS]
    for column, dataset in enumerate(DATASETS):
        group = scores[(scores.dataset == dataset) & (scores.protocol.isin(horizons))]
        for row, metric in enumerate(("MCC", "AUPRC")):
            axis = axes[row, column]
            for name, color in colors.items():
                points = group[group.condition == name].set_index("protocol").loc[horizons, metric]
                axis.plot(HORIZONS, points, color=color, marker="o", label=name, markersize=4)
            axis.set(xticks=HORIZONS, ylabel=metric if column == 0 else "",
                     xlabel="Future events" if row == 1 else "", title=dataset if row == 0 else "")
            axis.grid(alpha=.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False)
    fig.suptitle("Purchase prediction from event 10 on a fixed eligible cohort", fontsize=11)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"horizon_sensitivity.{suffix}", dpi=320)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    OUT.mkdir(exist_ok=True)
    results, cohorts, selected, runtimes = [], [], [], []
    for dataset in DATASETS:
        frame, columns, raw_parse, trace_column, trace_parse, basis, props, _ = audited_inputs(dataset)
        started = time.perf_counter()
        data, actions, _ = counts_and_prefix(frame, columns, raw_parse, trace_column, trace_parse,
                                              basis, props, with_transitions=True)
        runtimes.append(dict(dataset=dataset, matched_feature_prep_sec=time.perf_counter() - started))
        verify_test_replay(dataset, data["test"])
        train = data["train"]
        at_risk = train["length"] >= HORIZON
        prefix_train = {key: train[key][at_risk] for key in ("label", "transition_prefix")}
        full_columns, full_transitions = transition_selection(train, "transition_full", props)
        prefix_columns, prefix_transitions = transition_selection(prefix_train, "transition_prefix", props)
        selected.extend(dict(dataset=dataset, scope=scope, **row)
                        for scope, rows in (("full", full_transitions), ("prefix", prefix_transitions))
                        for row in rows)
        for future in (None, *HORIZONS):
            protocol = "full_observation" if future is None else f"next_{future}"
            tables = task_data(data, future)
            numerical = feature_suite(tables, full_columns if future is None else prefix_columns, future)
            for split in SPLITS:
                y = tables[split]["label"]
                cohorts.append(dict(dataset=dataset, protocol=protocol, split=split, n=len(y),
                                    positives=int(y.sum()), prevalence=y.mean()))
            results.extend(fit_controls(dataset, tables, protocol, actions, OUT, numerical=numerical))
    scores = pd.DataFrame(results)
    sizes = pd.DataFrame(cohorts)
    ci = uncertainty(scores)
    scores.to_csv(OUT / "metrics.csv", index=False)
    sizes.to_csv(OUT / "cohort_counts.csv", index=False)
    pd.DataFrame(selected).to_csv(OUT / "selected_transitions.csv", index=False)
    pd.DataFrame(runtimes).to_csv(OUT / "feature_prep_timing.csv", index=False)
    ci.to_csv(OUT / "paired_bootstrap.csv", index=False)
    for name, table in (("incremental_full_observation", scores[scores.protocol == "full_observation"]),
                        ("fixed_horizon", scores[scores.protocol != "full_observation"])):
        tex_table(table[["dataset", "protocol", "condition", "n_test", "MCC", "AUPRC"]],
                  OUT / f"{name}.tex", ".4f")
    plot_horizons(scores)
    meta = dict(seed=42, K=10, event_origin=10, future_event_counts=HORIZONS,
                transition_features=10, transition_min_train_present_and_absent=20,
                selection="Absolute training ever-purchase prevalence gap for binary adjacent-state atom pair presence. Prefix transition basis is held fixed across future horizons.",
                bootstrap_replicates=1500)
    (OUT / "iteration_protocol.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Verified and exported {len(scores)} matched fits.", flush=True)


if __name__ == "__main__":
    main()
