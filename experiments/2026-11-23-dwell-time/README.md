# Final fixed dwell-time supplement

Fixed matrix completed; `completion.json` and `final_validation.json` report success. No new test evaluation is permitted before all requested models and formulas are frozen.

Scope and frozen choices: [PROTOCOL.md](PROTOCOL.md), [config.json](config.json). Historical audit: [inventory_zh.md](inventory_zh.md), [historical_sources.json](historical_sources.json), [historical_aggregate_results.csv](historical_aggregate_results.csv). Observability checks: `unit_audit.json`, per-dataset `preparation.json`, and [audit_zh.md](audit_zh.md).

Run from this directory using the existing local Python3.10/private-source environment:

```powershell
python -u run.py
```

Do not launch a second runner while one is active. The initial execution used two independent mining processes and two selection processes, then `run.py --attached` to await their fixed results and run the neural/evaluation/report stages. Attached process IDs are invocation-specific; ordinary reruns use `run.py` without that argument. Completed pools, selection steps and neural epoch checkpoints are reused. Interrupted mining has no exact epoch-resume state: inspect a failure rather than silently replacing it. The historical experiment and manuscript remain untouched.

Dependencies and hardware are recorded in `environment.json`. Code reads the audited previous `2026-11-19-logical-history` modules and original private source layout from its config, including the original encoder sources in Downloads. Official LIBLINEAR2.50 is loaded from the previous local vendor directory. This is a reproducible local experiment tied to those hashed inputs, not a self-contained public-data package. Required packages include numpy, pandas, scipy, scikit-learn, torch, matplotlib, tabulate, threadpoolctl and psutil.

Final outputs: `main_results.csv`, `metrics.csv` (including every neural seed), `neural_mean_sd.csv`, `paired_differences.csv`, `selected_formulas.csv`, `selection_cost.csv`, `runtime.csv`, `paired_ap_differences.png`/`.svg`, `report_zh.md`, `manuscript_results_en.md`, `results_table.tex`, and final validation. Private timestamps/traces/IDs/predictions/models stay under ignored `private/`. Shareable code, settings and aggregate results are published to GitHub at the user’s request; private data, IDs, predictions and weights are excluded. The original paper is unchanged.


Post-run scientific and visual review passed (`delivery_review.json`). Read `report_zh.md` for the reviewed conclusions and `manuscript_results_en.md` for English insertion text. `results_table.tex` contains the linear table; `neural_results_table.tex` contains neural mean/sample-SD results. The opening synthesis in both reports was added after numerical report generation to disclose secondary-metric tradeoffs and seed heterogeneity; rerunning `report.py` regenerates the numerical report but replaces this editorial synthesis. No retraining or new evaluation is needed to use the delivered artifacts.


Electronics is historical-audit-only in this fixed supplement, as explicitly specified in the requested scope. See `inventory_zh.md` and `audit_zh.md` for the original hidden-purchase-gap leakage and remaining limitations in the patched historical results. No new Electronics model was trained, and its historical scores are not pooled into the new paired comparisons.
