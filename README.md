# LTL-based-Customer-Behavior-Prediction

## Residual-guided selection experiment

[Residual-guided LTLf formula selection — 2026-11-17](experiments/2026-11-17-residual-selection/README.md)

Electronics candidate-pool retraining and matched A/B/C comparisons across three datasets, with paired intervals, selection diagnostics and computational overhead. Full-observation AP improves over the historical Top-10 basis; event-10 gains are not consistent.

## Latest follow-up

[Fixed-origin neural and formula-source experiments — 2026-11-17](experiments/2026-11-17/README.md)

36 matched neural runs, DECLARE controls, Cosmetics pre-purchase formula relearning, formula audits, paired confidence intervals, and revised manuscript.

## Previous experiment snapshot

[Logic-guided session prediction — 2026-11-16](experiments/2026-11-16/README.md)

The previous snapshot contains the audited Coveo, Electronics, and Cosmetics experiments: 27 completed neural runs, count/transition controls, fixed-origin prediction, measured computational costs, frozen formula bases, and reproducible aggregate analysis.

```bash
cd experiments/2026-11-16
python -m pip install -r requirements-analysis.txt
python scripts/verify_export.py
python scripts/reproduce_figures.py
```

See the snapshot's [protocol](experiments/2026-11-16/PROTOCOL.md), [reproduction guide](experiments/2026-11-16/REPRODUCING.md), and [result map](experiments/2026-11-16/RESULTS_MAP.md). Fig.1 and the framework diagram include editable draw.io files.

The public artifact contains code and aggregate results. Raw clickstreams, individual session/user records, per-session predictions, split membership files, and model weights remain outside the repository.

## Historical notebook

[`PCS955experiment.ipynb`](PCS955experiment.ipynb) is retained as the original notebook. Use the dated snapshot for the current manuscript's experiment definitions and results.
