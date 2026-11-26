# Electronics extension to the fixed dwell-time supplement

Status: completed and validated on 2026-11-26. Six new Electronics LR conditions and nine neural fits are frozen and evaluated. Combined with the unchanged earlier experiments, this provides 18 LR conditions and 27 neural fits across three datasets.

This user-requested extension adds Electronics to the two completed datasets in ../2026-11-23-dwell-time. The earlier report remains a dated two-dataset artifact. This extension was specified after those results were observed and is exploratory.

Read the [Chinese report](report_zh.md), [English manuscript material](manuscript_results_en.md), and [Electronics data audit](audit_zh.md). Download the [three-dataset metrics](three_dataset_metrics.csv), [paired intervals](three_dataset_paired_differences.csv), [LR LaTeX table](results_table.tex), and [neural LaTeX table](neural_results_table.tex). The [figure](paired_ap_differences.png) summarizes the primary contrast.

All six primary M4−M3 AP intervals across datasets and model families include zero. Electronics LR AP difference is −0.03988 [−0.12317, +0.02018]; neural mean difference is −0.00125 [−0.00585, +0.00359]. This is not evidence of equivalence. Electronics has only 76 validation sessions (8 positives) and 171 test sessions (27 positives), with cross-split user overlap; secondary outcomes and seed variation require caution. See the reports for adverse as well as favorable outcomes.

Completion evidence: [completion.json](completion.json), [final_validation.json](final_validation.json), [aggregate provenance](aggregate_provenance.json), and [delivery review](delivery_review.json). Model/threshold freezing precedes test evaluation. No additional tuning or experiments were launched after evaluation.

Protocol: [PROTOCOL.md](PROTOCOL.md). Settings: [config.json](config.json). This run adds six LR conditions and nine neural fits, not a rerun of the other datasets. Historical leakage inventory remains in ../2026-11-23-dwell-time/inventory_zh.md.

Run locally with the verified private input layout and Python3.10 environment: `python -X utf8 -u run.py`. Do not launch duplicates while the runner is active. The prior native solver, learner and encoder sources remain read-only dependencies; hashes are recorded. Private raw histories/IDs/models/predictions are ignored by Git. Only aggregate artifacts and code are published.
