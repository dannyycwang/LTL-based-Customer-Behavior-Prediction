# Reproducing the experiments

There are two entry points: regenerate the published aggregate analysis, or run experiments with the corresponding local data. The first works from this repository alone.

## 1. Aggregate analysis

```bash
python -m pip install -r requirements-analysis.txt
python scripts/verify_export.py
python scripts/reproduce_figures.py
python scripts/diagrams.py
```

Generated files are written under `generated/`. The supplied `figures/` PDFs are the manuscript assets; the two `.drawio` files have editable shapes, labels, and connectors. To render regenerated diagrams, open them with draw.io Desktop and export cropped PDF/SVG. The original exports used draw.io 32.6.10. No editor binary is bundled.

The analysis dependencies are pinned to the export environment. Training also needs the packages in `requirements-training.txt` and a suitable PyTorch installation. Saved neural runs used Python 3.10, torch 2.11.0+cu128, CUDA 12.8, and an NVIDIA RTX 4060 Laptop GPU; exact versions are in `provenance/environment.json`.

## 2. Configure local inputs

Copy `configs/paths.example.json` to the ignored `paths.local.json`. Paths are absolute or relative to this snapshot directory. Point `audit_dir` and `work_dir` to an existing audited workspace, or use the default `private/outputs/research_audit` and `private/work` structure. The configuration selects files; it does not relax source-hash checks.

PowerShell:

```powershell
$env:LTLF_PATHS = (Resolve-Path paths.local.json).Path
python scripts/prepare_private_workspace.py
python scripts/check_private_inputs.py --hashes
```

Bash:

```bash
export LTLF_PATHS="$PWD/paths.local.json"
python scripts/prepare_private_workspace.py
python scripts/check_private_inputs.py --hashes
```

The workspace helper copies aggregate receipts only and retains existing receipts. Prepared CSVs, feature arrays, and raw source archives must be provided locally. The feature manifests contain historical paths normalized to `<workspace>/work/...`; these are resolved to `work_dir`. For older unmodified absolute paths on another machine, supply explicit `path_mappings` in the configuration.

Required layout under `work_dir`:

```text
electronics_rebuild/
  electronics_ready_noleak/
    electronics_sessions_ltl_ready.csv
    electronics_splits/seed_42/{train,val,test}.csv
  seed42_act_rel_features/{prefix_ltlf,full_ltlf,counts,counts_scaler}.npz
cosmetics_pre_purchase_seed42/
  {train,val,test}.csv
  features/
    prefix_ltlf/Z_BP_top10_cosmetics_dec.npz
    full_ltlf/Z_BP_top10_cosmetics_dec.npz
    counts/Z_BP_top20_cosmetics_dec.npz
    counts_scaler.npz
```

The `top20` name of the Cosmetics counts matrix denotes its 20 count/length coordinates; the logic basis still has K=10. Counts dimensions are 14/12/20 for Coveo/Electronics/Cosmetics. The Coveo enriched CSV is configured separately as `coveo_data`. It must match the saved source fingerprint.

For Coveo, reconstruct the notebook's initial split indices from that exact CSV:

```bash
python scripts/reconstruct_coveo_split.py
```

The neural runner's `protected_first_v1` protocol then keeps the verified formula pool in training. The original provenance-verification program is `replay/audit_coveo_split_overlap.py`; it additionally requires the private formula-training trace archive. The public split reconstruction does not repeat formula learning.

## 3. Preparation utilities

`models/electronics_ltlf_pipeline.py prepare` builds the initial Electronics sessions/splits, and `preprocessing/fix_electronics_dwell_leakage.py` creates the corrected ready directory. `preprocessing/prepare_cosmetics_dec_dataset.py` is the original Cosmetics preparation utility. Their `--help` output lists inputs and options. Once the audited source/split files and receipts are supplied, generate the downstream features with:

```bash
python replay/prepare_electronics_seed42.py
python replay/prepare_cosmetics_seed42.py
python replay/prepare_cosmetics_pre_purchase.py
```

The latter two commands cover historical Cosmetics and the corrected purchase-cutoff version. They also require the historical Cosmetics splits and original December archive in `work_dir`, plus provenance receipts. Checks enforce event alignment and the saved formula basis. Starting from the original source datasets requires reproducing the documented sampling and split assignments; downloading a dataset alone is not sufficient for byte-identical inputs.

`preprocessing/learn_cosmetics_tltlf_formulas.py` retains the historical formula-learning utility and its upstream TLTLf dependency. The current experiment suite uses the supplied frozen bases and does not invoke this utility. Source datasets and prior preparation settings are identified in the input manifests and the existing protocol documents. The complete original Colab runtime, private Drive files, and all original formula-training inputs are not part of this public artifact.

## 4. Neural prediction and optimization repeats

List the exact commands before execution:

```bash
python scripts/run_neural_suite.py --initializations --dry-run
```

Run the 15 main conditions, or include the 12 additional optimization runs:

```bash
python scripts/run_neural_suite.py
python scripts/run_neural_suite.py --initializations
```

The launcher uses 120 minutes per run by default and stops if a budget-limited run has no completion metrics. Increase `--max-minutes-per-run` and rerun to resume saved checkpoints. Completed runs are skipped. Fresh output directories are appropriate when changing model or experiment settings.

Individual examples:

```bash
python replay/run_coveo.py --seed 42 --condition full_ltlf --protocol protected_first_v1 --max-minutes 120
python replay/run_act_rel_seed42.py --dataset electronics --condition counts
python replay/run_act_rel_seed42.py --dataset cosmetics --observation pre_purchase --condition full_ltlf
```

The fixed bases are loaded directly from `formulas/`. Neural training and timing require CUDA. The original model files are preserved byte-for-byte; path adaptation in `replay/` is recorded in `provenance/portability.diff`. Historical runner fingerprints will therefore differ from the exported runner fingerprints, while model implementations and experiment choices remain explicit.

## 5. Linear controls, horizons, and cost

After the audited neural run inputs and outputs are available locally:

```bash
python replay/run_submission_controls.py
python replay/run_submission_iterations.py
python replay/benchmark_submission_runtime.py
python replay/benchmark_training_compute.py
```

The controls replay private split membership and verify feature/label alignment. Benchmarks load saved model weights and compare probabilities with saved predictions. They cannot execute from aggregate metrics alone. The exported aggregate results already contain the completed measurements, so running these commands is optional for inspecting the paper results.

To recalculate the current paired bootstrap from the original saved per-session predictions and user-cluster maps:

```bash
python scripts/recompute_bootstrap_from_private_predictions.py --workspace /path/to/original/audit/workspace
```

This workspace must contain `outputs/research_audit/runs/`, the linear prediction files, and `work/electronics_bootstrap_clusters.csv` / `work/cosmetics_bootstrap_clusters.csv`. Recomputed summaries go to `generated/statistics/`. The public `statistics/` contains the frozen summaries and aggregate replicate arrays for all displayed comparisons.

## Publication boundary

Public figure/table regeneration and aggregate verification are checked as part of this export. Full training needs the separately held data and saved preprocessing state; no new full training runs were executed for this export. Raw records, individual predictions, split IDs, vocabularies, and weights stay in ignored local directories. `MANIFEST.json` covers the public files only.
