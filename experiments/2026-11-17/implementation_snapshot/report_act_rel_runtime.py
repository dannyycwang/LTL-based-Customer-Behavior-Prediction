"""Report test feature construction and predictor timing with explicit boundaries."""
import json

import pandas as pd

from diagnose_secondary_seed42 import load_test
from report_coveo_seed42 import tex_table
from run_act_rel_seed42 import ROOT


def main():
    datasets = {
        "coveo": "Coveo",
        "electronics": "Electronics",
        "cosmetics": "Cosmetics (purchase-masked)",
        "cosmetics_pre_purchase": "Cosmetics (first-purchase cutoff)",
    }
    keys = {"Short+Prefix-LTLf": "prefix_ltlf", "Short+Counts": "counts", "Short+Full-LTLf": "full_ltlf"}
    rows = []
    for dataset, label in datasets.items():
        directory = ROOT / f"{dataset}_seed42_comparison"
        summary = pd.read_csv(directory / "run_summary.csv")
        if len(summary) != 5:
            raise ValueError(f"Expected five completed runs for {label}")
        if dataset == "coveo":
            info = json.loads((directory / "formula_diagnostic_metadata.json").read_text(encoding="utf-8"))
            times = {**info["test_feature_seconds_from_parsed_traces"],
                     "symbolic_parse_sec": info["test_symbolic_parse_sec"]}
        else:
            *_, times = load_test(dataset)
        for run in summary.itertuples(index=False):
            key = keys.get(run.condition)
            parse = times["symbolic_parse_sec"] if key else 0.0
            features = times[key] if key else 0.0
            predictor = run.test_inference_precomputed_sec if dataset == "coveo" else run.inference_precomputed_features_sec
            rows.append(dict(dataset=label, condition=run.condition, n_test=run.n,
                             test_symbolic_parse_sec=parse, test_symbolic_feature_sec=features,
                             gpu_inference_precomputed_sec=predictor,
                             sequential_component_sum_sec=parse + features + predictor,
                             all_split_feature_construction_sec=run.feature_construction_all_splits_sec,
                             training_validation_sec=run.train_including_validation_sec))
    out = ROOT / "act_rel_seed42_summary"
    data = pd.DataFrame(rows)
    data.to_csv(out / "runtime.csv", index=False)
    for latex, columns in {
        "runtime_training.tex": ["dataset", "condition", "all_split_feature_construction_sec", "training_validation_sec"],
        "runtime_test.tex": ["dataset", "condition", "n_test", "test_symbolic_parse_sec",
                             "test_symbolic_feature_sec", "gpu_inference_precomputed_sec", "sequential_component_sum_sec"],
    }.items():
        subset = data[columns].rename(columns={
            "dataset": "Dataset", "condition": "Condition", "n_test": "Test n",
            "test_symbolic_parse_sec": "Parse (s)", "test_symbolic_feature_sec": "Features (s)",
            "gpu_inference_precomputed_sec": "GPU (s)", "sequential_component_sum_sec": "Sum (s)",
            "all_split_feature_construction_sec": "Features (s)", "training_validation_sec": "Train + val (s)",
        })
        tex_table(subset, out / latex, ".2f")
    print(f"Exported {len(data)} runtime rows.", flush=True)


if __name__ == "__main__":
    main()
