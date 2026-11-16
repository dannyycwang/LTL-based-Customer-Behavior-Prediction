# Manuscript and experiment map

| Manuscript item | Artifact source | Regeneration |
|---|---|---|
| Fig.1, clickstreams to pattern vectors | `figures/fig1_patterns.drawio` | `scripts/diagrams.py`; export with draw.io |
| Fig.2, observation boundary and two branches | `figures/fig2_framework.drawio` | Same |
| Fig.3, scope gains by history length | `statistics/contrasts.csv`, neural short/long strata | `scripts/reproduce_figures.py` |
| Fig.4, horizon comparison | `statistics/*_next_*_replicates.npz` | Same; paired contrasts |
| Fig.5, cost decomposition | `results/submission_controls_seed42/*_repetitions.csv` | Same; mean±sample SD |
| Table 2, cohort definitions | `results/submission_iterations_seed42/cohort_counts.csv` | Same |
| Table 3, neural MCC/AP | `statistics/estimates.csv`, `neural_all` | Same |
| Table 4, linear representation comparisons | `statistics/estimates.csv`, `full_observation`/`next_10` | Same |
| Table 5, prediction and computation | `training_runtime.csv`, `inference_runtime_repetitions.csv`, neural estimates | Same |
| Optimization variation | `results/submission_controls_seed42/neural_initialization_deltas.csv` and 27 run folders | `replay/report_submission_controls.py` requires private predictions |
| Length/action/count decomposition | `results/submission_controls_seed42/metrics.csv`, `cosmetics_metrics.csv` | `replay/run_submission_controls.py` requires private inputs |

The literature taxonomy (Table 1) is not an experiment and is not part of this export. Figure filenames retain their original internal identifiers; the thematic manuscript does not use RQ headings.

Each of the 27 run folders contains aggregate metrics and a configuration receipt. Coveo has `epochs.jsonl`; Electronics and Cosmetics have `epochs.json`. Training-time values refer to the completed runs, not a synthetic epoch-time projection. Hashes in historical configurations identify the exact source files used at execution time; `provenance/source_files.json` also records the exported source hashes and the path-portability changes.
