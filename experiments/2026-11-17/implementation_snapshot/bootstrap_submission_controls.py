"""Paired finite-test uncertainty for the saved seed-42 neural models."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from report_coveo_seed42 import tex_table
from run_act_rel_seed42 import DEFAULT_REBUILD, PRE_PURCHASE_DATA, PRE_PURCHASE_PROTOCOL, PROTOCOL, ROOT


OUT = ROOT / "submission_controls_seed42"
GROUPS = {
    "Coveo": "protected_first_v1_seed42",
    "Electronics": f"electronics_{PROTOCOL}",
    "Cosmetics": f"cosmetics_{PRE_PURCHASE_PROTOCOL}",
}
CLUSTERS = {
    "Electronics": DEFAULT_REBUILD / "electronics_ready_noleak" / "electronics_splits" / "seed_42" / "test.csv",
    "Cosmetics": PRE_PURCHASE_DATA / "test.csv",
}
CONTRASTS = (("full_ltlf", "counts", "all"), ("full_ltlf", "prefix_ltlf", "L>10"),
             ("counts", "full_raw", "all"))


def score_function(label, probability, threshold):
    order = np.argsort(-probability, kind="stable")
    sorted_p = probability[order]
    y_order = label[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_p) != 0), len(label) - 1]
    decision = probability >= threshold
    positive_decision = label * decision

    def scores(weight):
        positive = np.dot(weight, label)
        predicted = np.dot(weight, decision)
        true_positive = np.dot(weight, positive_decision)
        total = weight.sum()
        denominator = float(positive) * float(total - positive) * float(predicted) * float(total - predicted)
        mcc = (total * true_positive - positive * predicted) / np.sqrt(denominator) if denominator > 0 else 0.
        ordered = weight[order]
        tp = np.cumsum(ordered * y_order)[ends]
        samples = np.cumsum(ordered)[ends]
        precision = np.divide(tp, samples, out=np.zeros(len(samples)), where=samples > 0)
        ap = np.dot(np.diff(tp, prepend=0), precision) / positive if positive else np.nan
        return np.array([mcc, ap])

    return scores


def load_predictions(dataset, conditions):
    result = {}
    reference = None
    for condition in conditions:
        path = ROOT / "runs" / f"{GROUPS[dataset]}_{condition}" / "predictions.csv"
        pred = pd.read_csv(path, dtype={"session_id": str}).set_index("session_id", verify_integrity=True)
        keys = pred[["label", "length"]]
        if reference is None:
            reference = keys
        elif not reference.equals(keys):
            raise ValueError(f"Misaligned predictions: {dataset}/{condition}")
        if pred.threshold.nunique() != 1:
            raise ValueError(f"Multiple thresholds: {dataset}/{condition}")
        result[condition] = pred
    return result


def cluster_ids(dataset, sessions):
    if dataset not in CLUSTERS:
        return np.arange(len(sessions)), "session"
    users = pd.read_csv(CLUSTERS[dataset], usecols=["session_id_hash", "user_id"], dtype=str)
    aligned = users.set_index("session_id_hash", verify_integrity=True).loc[sessions]
    if aligned.user_id.isna().any():
        raise ValueError(f"Missing user ID: {dataset}")
    return pd.factorize(aligned.user_id)[0], "user"


def paired_interval(label, probability_a, probability_b, threshold_a, threshold_b, groups, replicates, rng):
    n_clusters = int(groups.max()) + 1
    score_a = score_function(label, probability_a, threshold_a)
    score_b = score_function(label, probability_b, threshold_b)
    unweighted = np.ones(len(label), dtype=np.int64)
    point = score_a(unweighted) - score_b(unweighted)
    deltas = np.empty((replicates, 2))
    for i in range(replicates):
        sampled = rng.integers(n_clusters, size=n_clusters)
        weight = np.bincount(sampled, minlength=n_clusters)[groups]
        deltas[i] = score_a(weight) - score_b(weight)
    lower, upper = np.nanquantile(deltas, [.025, .975], axis=0)
    return point, lower, upper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=1500)
    args = parser.parse_args()
    if args.replicates < 100:
        parser.error("Use at least 100 replicates.")
    rng = np.random.default_rng(20261112)
    results = []
    for dataset in GROUPS:
        predictions = load_predictions(dataset, sorted({key for left, right, _ in CONTRASTS for key in (left, right)}))
        for left, right, stratum in CONTRASTS:
            a, b = predictions[left], predictions[right]
            if stratum == "L>10":
                a, b = a[a.length > 10], b[b.length > 10]
            groups, unit = cluster_ids(dataset, a.index)
            n_clusters = int(groups.max()) + 1
            point, lower, upper = paired_interval(
                a.label.to_numpy(dtype=np.int64), a.probability.to_numpy(dtype=np.float32),
                b.probability.to_numpy(dtype=np.float32), a.threshold.iloc[0], b.threshold.iloc[0],
                groups, args.replicates, rng,
            )
            for j, metric in enumerate(("MCC", "AUPRC")):
                results.append(dict(dataset=dataset, contrast=f"{left} minus {right}", stratum=stratum,
                                    n_sessions=len(a), unit=unit, n_units=n_clusters, replicates=args.replicates,
                                    metric=metric, delta=point[j], lower_95=lower[j], upper_95=upper[j]))
        print("Paired bootstrap complete:", dataset, flush=True)
    out = pd.DataFrame(results)
    OUT.mkdir(exist_ok=True)
    out.to_csv(OUT / "neural_paired_bootstrap.csv", index=False)
    tex_table(out[["dataset", "contrast", "stratum", "unit", "metric", "delta", "lower_95", "upper_95"]],
              OUT / "neural_paired_bootstrap.tex", ".4f")


if __name__ == "__main__":
    main()
