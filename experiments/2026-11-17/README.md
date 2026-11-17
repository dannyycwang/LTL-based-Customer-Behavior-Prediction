# Fixed-origin follow-up — published 2026-11-17

Run ID: `next_round_20261116_v1`. This adds 36 completed neural runs (3 datasets x 4 conditions x 3 seeds), 14 linear controls, two Cosmetics basis-source fits, 30 frozen-formula audits and paired bootstrap intervals. The main frozen bases and original splits were retained; negative results were not replaced.

- [English supplement](results/experiment_supplement.pdf)
- [Compiled recovered-source manuscript](paper_recovered_revision/clean.pdf)
- [Latest-PDF replacement text](results/latest_pdf_replacements.tex)
- [Protocol](protocol.json)
- [Neural results](results/neural_mean_sample_sd.csv)
- [Template comparison](results/linear_metrics.csv)
- [Source comparison](cosmetics_source/comparison/paired_differences.csv)
- [Completion and limitations](results/completion_status.json)

LTLf improves count-augmented linear models, but has no consistent incremental neural benefit. Comparisons do not establish automatic-learning superiority over transitions or DECLARE templates. The Cosmetics pre-purchase source contrast has intervals crossing zero. No overall training speedup is claimed.

## Reproduction scope

The Python files archive the executed local implementation, including local paths and dependencies. They are not a turnkey standalone training package. `common.py` expects the original project layout; `implementation_snapshot/` contains inherited source dependencies. Recreate the private data/project layout and adjust local paths before running the commands in `README_先讀.txt`. The previous [snapshot reproduction guide](../2026-11-16/REPRODUCING.md) documents the earlier data pipeline. Changing paths does not authorize changing split, basis, labels or selection settings.

The public snapshot contains aggregate results, per-run configurations/epoch logs, formulas, code, and manuscript assets. Raw traces, session/user membership, per-record predictions, feature caches, model weights and the local review ZIP are deliberately excluded. The Chinese delivery note describes the complete local delivery; paths to those private artifacts are not GitHub downloads.

The manuscript is based on a recovered earlier LaTeX version, not the exact latest uploaded PDF source. It has 11 pages and the supplement has 7; compilation and visual checks passed. Original experiments and the historical notebook remain unchanged.
