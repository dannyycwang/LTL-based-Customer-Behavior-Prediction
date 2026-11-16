# Cosmetics first-purchase cutoff, Act+Rel seed 42

Positive histories end immediately before their first purchase. Raw events are
ordered by timestamp and original file row, following the historical notebook.
Negative histories keep all observed events. Sessions with no observable event
are excluded before training or evaluation. The original user-disjoint split
assignments are preserved and negatives are kept without resampling. The
remaining test cohort has 25,836 sessions and 5,140 positives.

The raw replay matches all 130,315 historical session IDs, labels, complete
event counts, retained action sequences and product/category sequences. The
Act+Rel propositions use current and past fields. The verified prefix can
therefore be sliced from the saved symbolic representation. There is no dwell
feature in these conditions.

The same ten existing seed-42 BP formulas are used for both observation
protocols. They were fit and ranked with historical purchase-masked traces
from train and validation respectively. Those samples included 526 fitting
and 259 ranking sessions whose histories extended past first purchase.
The new run therefore measures transfer of the existing basis to inputs cut
before purchase. The downstream models and Counts scaler are newly fitted
under the cutoff. Test sessions and users remain outside the original
formula-fitting and ranking pools.

The five models use the original Cosmetics predictor, with first-10 raw input
for the four short conditions. The historical Full raw cap of 200 is kept
and truncates 21 test sessions. Full-LTLf and Counts use the entire observed
symbolic trace. Validation MCC selects checkpoints, with a fixed threshold of
0.5 for validation and test. Epoch cap is 25, patience is 5, train batch size
is 512 and evaluation batch size is 1024. MCC is primary and AUPRC is average
precision. Comparisons are descriptive for a single seed.

The observation-protocol comparison evaluates both trained model sets on the
same 25,836 sessions with the same labels. The observation cutoff changes
training inputs as well as test inputs. The comparison includes retraining
and the training/validation eligibility changes. It therefore measures the
difference between the two pipelines under a common test cohort. The
purchase-relative cutoff answers an observation-policy question. Evaluation
at a fixed online time horizon would require its own cohort definition.

## Reproduction

Run `prepare_cosmetics_pre_purchase.py` and
`diagnose_secondary_seed42.py --dataset cosmetics_pre_purchase` from
`outputs/research_audit`. Run `run_act_rel_seed42.py --dataset cosmetics
--observation pre_purchase --condition full_raw --max-minutes 60` for Full raw,
or substitute one of the other four conditions. The per-epoch checkpoints
include optimizer and RNG state. Run `report_cosmetics_pre_purchase.py` for
verified metrics and the matched-cohort comparison.
