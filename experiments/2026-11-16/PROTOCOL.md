# Experimental protocol

## Observation and representation

All short raw branches use the **first ten events**, with action and product/category IDs. Symbolic features use Act+Rel propositions. The formula basis is fixed at **K=10**: the first ten entries of saved ranked 20-formula Coveo/Cosmetics bases, and the saved ten-formula Electronics basis. Prefix and full checking share the same basis within a dataset. Counts include proposition frequencies and length and are scaled using training data.

For full observation, positive histories end immediately before first purchase; negative histories retain their recorded events. The target concerns purchase in the original session. The primary Cosmetics results use the corrected event parser and pre-purchase histories; 34 empty positive test histories are excluded. Historical Cosmetics comparisons remain documented in the aggregate evidence and are not mixed into the main table.

Full raw caps are 200, 512, and 200 events for Coveo, Electronics, and Cosmetics; these truncate zero, one, and 21 test histories. Symbolic summaries retain the full available history.

## Splits and frozen formulas

The downstream split seed is 42. Coveo's protected split keeps all 20,000 verified formula-fitting traces in training. Formula fitting/ranking used training traces for Coveo and Electronics. Cosmetics fitted on training traces and ranked on validation traces; its historical source includes 526 fitting and 259 ranking histories extending beyond first purchase. Its saved basis is transferred to the purchase-cutoff task. Test sessions remain outside the formula pools.

Cosmetics is user-disjoint. Electronics evaluates unseen sessions with 2,784 users shared between training and test. Coveo user IDs are unavailable. Electronics was rebuilt from the source dataset and checked against saved notebook aggregate evidence; original historical split-file identity was unavailable. Input fingerprints and the scope of the reconstruction are preserved in `provenance/input_manifests/`.

Formula learning is not repeated for downstream initialization seeds 43 and 44. The additional runs change optimization randomness while retaining the same split and basis.

## Predictors and temporal controls

The neural conditions use the original dataset-specific Transformer implementations in `models/`. Coveo trains 25 epochs, selects by validation F1 at threshold .5, then selects a validation-F1 decision threshold. Electronics uses at most 20 epochs, patience 4, validation AP selection, and threshold .5. Cosmetics uses at most 25 epochs, patience 5, validation MCC selection, and threshold .5. Detailed batch sizes and parameter counts are in each saved run configuration.

Shared linear predictors use L2 logistic regression (C=1), position-wise raw-prefix tokens, train-standardized numeric features, and a validation-MCC threshold. Logic and transitions each add ten Boolean features. Adjacent proposition-pair transitions are ranked by absolute training purchase/non-purchase prevalence difference after requiring at least 20 present and 20 absent training sessions. The combined condition adds both sets.

## Fixed prediction origin

Sessions must reach event 10 before purchase. Every feature uses events 1–10. The target is first purchase in the next H=5,10,20 recognized events, counting the purchase event itself. A later purchase or an ended session without purchase within the horizon receives zero. The same eligible cohort is used across horizons. Formula and prefix-transition bases remain fixed, with selection based on the original session-purchase target.

## Uncertainty and timing

The current presentation uses the sample SD of 1,500 bootstrap metric estimates as **one SE**; it is not divided by the square root of 1,500. Contrasts use shared resamples. Resampling units are users for Electronics/Cosmetics and sessions for Coveo; models, thresholds, splits, and bases remain fixed. The `lower_95`/`upper_95` audit columns are percentile bounds, not the displayed ±SE. Training-seed and timing variation use sample SD instead.

Controlled Coveo updates use three repeats, identical precomputed-input minibatches, five warm-up updates and forty measured updates. A repeat starts from the saved checkpoint, and condition order rotates. The timer includes collation, transfers, forward/backward computation, clipping, optimizer work, and CUDA synchronization.

Inference uses one warm-up and three measured passes. It begins with loaded session fields and includes raw packing, symbolic parsing/checking or counting, collation, transfers, and prediction. Model loading, disk I/O, upstream event abstraction, and historical formula learning are outside this boundary. Training totals include validation/checkpoints and begin after feature precomputation. Epoch counts accompany training times. GPU memory is peak PyTorch CUDA allocation including model weights.
