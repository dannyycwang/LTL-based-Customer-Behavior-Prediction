# Submission controls, seed-42 Act+Rel splits

All experiments reuse K=10 saved formulas per dataset. Coveo uses the first
10 formulas in the 20-formula APC basis. Electronics uses its archived
10-formula BP basis. Cosmetics uses the first 10 formulas in its corrected
20-formula BP basis. The formula sets differ across datasets. The prefix and
full checking conditions within each dataset share one formula set. The
original formula-fitting and ranking targets concern purchase at some point
in the session. The fixed event-horizon experiment reuses this basis.

## Count controls on full observed histories

The CPU controls use an L2-regularized logistic model with C=1 and the
same position-wise one-hot action, product and category tokens for the first
10 events. Each added numeric feature is standardized with training data.
The length baseline uses log(1 + observed length). Action proportions divide
the count of each action by the observed length. Action counts sum exactly
to length in these traces, so raw action counts also carry length information.
Proportions remove the explicit count scale while retaining possible indirect
length cues. The Act+Rel counts use the original proposition vocabulary. The
relation atoms can summarize earlier activity even though their counts omit
event positions. Short histories have explicit right-padding tokens.

The full-observation controls use the complete audited pre-purchase sequence
for added full-trace features. This experiment addresses the predictive value
of the representation when later events are available to the feature branch.
The linear architecture and validation threshold differ from the original
Transformer experiments. Interpret the contrasts within each experiment.

## Fixed event horizon

The eligible cohort consists of sessions with at least 10 observed events
before any purchase. Predict immediately after observed event 10. The target
is a first purchase in events 11–20 of the recognized event stream, counting
the purchase event itself. For positive original labels, first-purchase
position is observed pre-purchase length plus one. A purchase after position
20 receives target 0. A session that ends before position 20 without purchase
also receives target 0. This is an event-based horizon on the historical
sample, which uses case-control negative sampling. Report the cohort sizes and
class prevalences alongside scores. Its event clock can differ from a fixed
wall-clock horizon or a deployment cohort.

Every input field is restricted to the same ten observed events. The feature
suites are raw positional tokens, raw plus action counts, raw plus Act+Rel
counts, and raw plus Prefix-LTLf. Raw and symbolic fields were checked for
length agreement and the recalculated test formula matrices exactly match
the audited ones. Thresholds maximize validation MCC over a fixed grid.
No test labels enter the model fit or threshold choice.

## Uncertainty

The paired percentile intervals use 1,500 resamples of the held-out test
population. Both models in each contrast use identical resamples. The unit
is user for Electronics and Cosmetics, session for Coveo where user IDs are
unavailable. Thresholds and fitted models stay fixed. The intervals describe
test-sampling uncertainty conditional on the training data and fitted models.
They are descriptive and use individual 95% intervals.

The downstream initialization experiments fix data, split seed 42, formula
basis and hyperparameters and rerun Counts and Full-LTLf with training seeds
43 and 44. This changes parameter initialization, batch order and stochastic
training operations. Variation across training seeds pertains to the same
fixed evaluation split and formula set.

## Compute measurements

The matched inference benchmark loads the same saved seed-42 models and
test-session ordering for all five conditions in each dataset. One warm-up
and three measured passes are used on the same GPU. It verifies replayed
probabilities against the saved predictions. The measured interval includes
raw field encoding, symbolic parsing and checking or counting, batch
collation, transfers, and model prediction. CUDA is synchronized before and
after each pass. Disk I/O, upstream event abstraction, initial model loading
and learning the formula basis are outside this interval. The memory column
reports peak PyTorch CUDA allocation within a pass, including model weights.

Training epoch durations use observed training plus validation time divided
by completed epochs. Early stopping and ambient workload can vary across
runs. The controlled Coveo update benchmark additionally replays the same
precomputed-input minibatches across Full raw, Short + Counts, and Short +
Full-LTLf. It includes collation, transfers, forward/backward, clipping and
optimizer updates. Validation and feature preparation are measured separately.
Within each of three repeats, the model begins from the saved checkpoint and
updates an in-memory copy on the same batches, with five warm-up batches and
40 measured batches. Condition order rotates across repeats.
