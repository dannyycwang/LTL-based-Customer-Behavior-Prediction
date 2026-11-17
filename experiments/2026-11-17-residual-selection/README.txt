Residual-guided LTLf experiment

This directory contains new artifacts only. The manuscript, historical models,
data splits and original results are read-only inputs.

Electronics retraining completed in runs/electronics_retrain_20261117_093115.
The training recipe uses the historical notebook's functions and configuration:
seed 42, 4000 positives + 4000 negatives from audited training, last 10 events,
Act+Rel (BP), network length 10, AdamW lr .01, batch 100, 50 epochs.
Training JSON was reconstructed and compared with the previously audited JSON.
No validation or test labels were used to train/rank this candidate export.
The model is newly trained, not claimed identical to the lost historical model.
250 extracted formulas have 122 distinct fitting-sample satisfaction signatures.
semantic_verification.json records independent finite-trace checker validation.

Entrypoints (PowerShell, Python 3.10 environment used for the original audit):
  $env:PYTHONIOENCODING='utf-8'
  python residual_guided_ltlf/retrain_electronics.py
  python residual_guided_ltlf/test_selection.py
  python residual_guided_ltlf/run_residual.py --dataset Coveo --out NEW_OUTPUT
  python residual_guided_ltlf/run_residual.py --dataset Cosmetics --out NEW_OUTPUT
  python residual_guided_ltlf/run_residual.py --dataset Electronics --electronics CANDIDATE_CSV --out NEW_OUTPUT

run_residual.py imports the original audited pipeline from:
C:/Users/ycw/Documents/Codex/2026-11-10/new-chat/outputs/research_audit
It verifies source hashes and original feature replay before fitting. Original
dependencies/data must remain available there; this is a local reproducible
extension, not a standalone redistributable dataset package.

Conditions:
A = raw prefix + Counts + the original selected ten Transitions.
B = A + the historical Top-10 formulas.
C = A + ten residual-guided formulas from the available candidate export.
D (Electronics only) = A + the retrained pool's historical-style standalone
ranking Top-10, using the same 8000 fitting traces. This helps distinguish new
candidate generation from selection strategy. Original formulas absent from
the new export are retained for B/diagnostics but are not eligible for C.

Selection uses a C=1 liblinear logistic model conditioned on Counts+Transitions,
with no raw prefix. The final A/B/C/D models all include the same raw prefix.
At each step choose the largest validation log-loss reduction, refit, repeat to
K=10 even if later gains are nonpositive. Select separately for full observation
and event-10 next-ten-event prediction. Target-specific reselection differs from
the historical frozen formula basis and must be disclosed in interpretation.
Exclude training-constant candidates and keep the first exported representative
of exact training satisfaction duplicates. No test/validation labels determine
deduplication. This prevents duplicated columns from masquerading as information
through changed effective L2 regularization. Record all excluded candidates.

For efficiency independent candidate fits use four threads, with underlying
thread pools limited to one thread. Explicit random_state=42 and candidate order
make tie handling deterministic. Exact train/dev feature duplicates may share
computed loss values; this cache does not choose or remove candidates.

No downstream hyperparameter search: retain the audited C=1 classifier and fixed
validation MCC threshold grid. Test probabilities are evaluated only after the
formula set is frozen. Bootstrap uses 1500 common resamples across conditions:
sessions for Coveo, users for Electronics/Cosmetics. Report conditional sampling
uncertainty, not uncertainty over formula mining or model selection.
One audited split is retained. Additional logistic seeds are not meaningful
independent data replications and are not presented as such.

Every completed dataset directory contains provenance, selection traces,
per-candidate development diagnostics, final predictions/metrics, selected
formulas, bootstrap intervals, and correlations. Some runs additionally cache
features for verified reuse; --feature-cache requires matching hashes/IDs.
check_replay.py compares A/B with historical predictions without selecting on
test performance. redundancy_report.py finds training-only witnesses with equal
Counts+selected Transitions but differing formula values. Such witnesses do NOT
prove information absent from raw events, or useful generalization by themselves.

Interrupted development runs:
cosmetics_v1: initial serial selection, stopped before completing step 1.
cosmetics_v2: parallel selector before duplicate-column guard, interrupted.
electronics_v1: feature cache and initial pre-guard selection, interrupted.
These runs are not final experiments and must not be pooled into reported results.
The synthetic test output is software validation, not research evidence.

Completed final runs: coveo_v2, electronics_v2, cosmetics_v3.
Coveo v1 is a preparation cache, not a final fitted comparison. Coveo v2 resumed
its frozen full-observation selection for memoized duplicate diagnostic fits;
selection decisions were retained. No test results informed this optimization.
Strong witness files now match Counts and ALL adjacent occurrence bits, not just
the ten selected bits. Counts/ALL-adjacency matching still does not mean identical
raw events or identical transition frequencies.
