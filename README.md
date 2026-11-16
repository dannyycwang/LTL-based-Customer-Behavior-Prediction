# LTL-based-Customer-Behavior-Prediction

## Current experiment snapshot

[Logic-guided session prediction — 2026-11-16](experiments/2026-11-16/README.md)

The current snapshot contains the audited Coveo, Electronics, and Cosmetics experiments: 27 completed neural runs, count/transition controls, fixed-origin prediction, measured computational costs, frozen formula bases, and reproducible aggregate analysis.

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
