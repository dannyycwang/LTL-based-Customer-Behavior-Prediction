# Manuscript-ready exploratory results

## Results and discussion for manuscript insertion

Adding nonlogical timing features to the linear raw-plus-summary baseline improved AP on Coveo by 0.01658 (95% paired percentile interval, 0.00604 to 0.02678), whereas the Cosmetics difference was 0.00505 (−0.00983 to 0.02073). Corresponding MCC and Brier difference intervals included zero on both datasets; the Cosmetics Brier point estimate increased by 0.00313. Thus, evidence for ordinary timing features was dataset- and metric-dependent.

Conditional on the same timing inputs, expanding the learned LTLf vocabulary from Act+Rel to Act+Rel+Dwell did not establish an additional AP benefit. The linear M4−M3 differences were 0.00093 (−0.00203 to 0.00375) on Coveo and 0.00108 (−0.00347 to 0.00500) on Cosmetics. The corresponding neural seed-average differences were −0.00114 (−0.00454 to 0.00252) and 0.00055 (−0.00867 to 0.00948). AP intervals for M4−M2 also included zero in both model families, and the linear comparison with DECLARE did not establish learned-LTLf superiority. These findings do not demonstrate equivalence or absence of an effect.

Secondary outcomes were mixed. On Cosmetics, neural M4−M3 reduced mean Brier score by 0.01345 (−0.01563 to −0.01112), but seed-specific differences were −0.03901, −0.00723, and 0.00590. The AP differences for those seeds were 0.01171, 0.01147, and −0.02154 (sample SD, 0.01913); on Coveo they were −0.00340, −0.00063, and 0.00060 (sample SD, 0.00205). All six seed-specific primary AP intervals included zero. Cosmetics neural M4−M2 improved MCC by 0.05156 (0.02670 to 0.07667) and Brier by −0.01036 (−0.01266 to −0.00821). M3−M2 likewise improved MCC by 0.04241 (0.02232 to 0.06515), while worsening Brier by 0.00308 (0.00144 to 0.00463). These exploratory secondary findings do not replace the prespecified AP conclusion. Shared cluster resampling conditions on the three trained models and does not capture uncertainty over optimization seeds.

Four of the ten selected M4 formulas on each dataset referenced known-duration buckets; two additional Cosmetics formulas used the unknown-gap marker. All selected formulas contained temporal syntax. None exactly matched the enumerated univariate dwell-count threshold rules on training data, although one Coveo duration formula agreed with such a rule on 99.9016% of training records. This limited diagnostic does not establish irreducible temporal information or causal interpretation. The main comparison concerns the complete vocabulary, learning, and selection pipelines, not an isolated dwell operator. Formula selection was substantially more expensive for DECLARE: approximately 19.91 hours on Coveo and 7.91 hours on Cosmetics, compared with 4.35 and 0.87 hours for M4. These invocation-specific costs include candidate refits and machine contention; cached-input neural training is not an end-to-end speed comparison.


We evaluated a fixed event-20 cohort with a ten-event raw window and a ten-event first-purchase horizon on Coveo and Cosmetics. Electronics was included only in the historical audit. Outgoing event gaps were constructed after truncating the observable prefix; the terminal gap was explicitly unknown. Bucket thresholds and numeric transformations were fitted on training data only.

For Coveo, the linear M2-M0 AP difference was +0.01658 [+0.00604, +0.02678].
For Coveo, the linear M4-M3 AP difference was +0.00093 [-0.00203, +0.00375].
For Coveo, the linear M4-M2 AP difference was +0.00221 [-0.00032, +0.00457].
For Coveo, the linear M4-M5 AP difference was -0.00005 [-0.00646, +0.00576].
The neural mean paired AP difference (M4−M3) across optimization seeds 42–44 was -0.00114 [-0.00454, +0.00252]; all seeds shared the same cluster resamples.
The neural seed-average M4-M2 AP difference was -0.00195 [-0.00800, +0.00415].
The neural seed-average M3-M2 AP difference was -0.00081 [-0.00722, +0.00565].
For the LR primary contrast, the paired secondary differences were MCC -0.00645 [-0.01319, +0.00037]; Brier +0.00007 [-0.00032, +0.00043]. Higher MCC and lower Brier are preferable.
For the NN_mean primary contrast, the paired secondary differences were MCC -0.00117 [-0.00603, +0.00384]; Brier -0.00012 [-0.00063, +0.00040]. Higher MCC and lower Brier are preferable.
For Cosmetics, the linear M2-M0 AP difference was +0.00505 [-0.00983, +0.02073].
For Cosmetics, the linear M4-M3 AP difference was +0.00108 [-0.00347, +0.00500].
For Cosmetics, the linear M4-M2 AP difference was +0.00125 [-0.00332, +0.00522].
For Cosmetics, the linear M4-M5 AP difference was +0.00105 [-0.00597, +0.00787].
The neural mean paired AP difference (M4−M3) across optimization seeds 42–44 was +0.00055 [-0.00867, +0.00948]; all seeds shared the same cluster resamples.
The neural seed-average M4-M2 AP difference was +0.00631 [-0.00271, +0.01565].
The neural seed-average M3-M2 AP difference was +0.00576 [-0.00254, +0.01442].
For the LR primary contrast, the paired secondary differences were MCC +0.01717 [-0.00305, +0.03524]; Brier -0.00046 [-0.00168, +0.00087]. Higher MCC and lower Brier are preferable.
For the NN_mean primary contrast, the paired secondary differences were MCC +0.00915 [-0.00929, +0.02705]; Brier -0.01345 [-0.01563, -0.01112]. Higher MCC and lower Brier are preferable.

The primary comparison contrasts two vocabulary and formula-learning pipelines, not the causal effect of a single dwell operator. All comparisons are exploratory because the test set previously participated in research development. Intervals condition on fitted models, formulas, and thresholds and exclude mining uncertainty. Failure to exclude zero is not evidence of equivalence. Formulas selected with logistic regression were transferred unchanged to the neural models, limiting architecture-level conclusions.

| dataset   | condition   |       AP |    Brier |      MCC |   selected_logic_features |
|:----------|:------------|---------:|---------:|---------:|--------------------------:|
| Cosmetics | M0          | 0.404481 | 0.207781 | 0.182219 |                         0 |
| Cosmetics | M1          | 0.404974 | 0.207691 | 0.172627 |                         5 |
| Cosmetics | M2          | 0.409534 | 0.210913 | 0.213567 |                         0 |
| Cosmetics | M3          | 0.409709 | 0.210732 | 0.202495 |                         8 |
| Cosmetics | M4          | 0.410786 | 0.21027  | 0.219662 |                        10 |
| Cosmetics | M5          | 0.40974  | 0.211012 | 0.204988 |                        10 |
| Coveo     | M0          | 0.505594 | 0.127658 | 0.468111 |                         0 |
| Coveo     | M1          | 0.509207 | 0.127164 | 0.467374 |                        10 |
| Coveo     | M2          | 0.522178 | 0.126446 | 0.464535 |                         0 |
| Coveo     | M3          | 0.523457 | 0.126201 | 0.46658  |                        10 |
| Coveo     | M4          | 0.524391 | 0.126271 | 0.460127 |                        10 |
| Coveo     | M5          | 0.524442 | 0.125902 | 0.466359 |                        10 |

Neural performance (mean and sample SD across three optimization seeds):

| dataset   | condition   | metric   |     mean |        std |
|:----------|:------------|:---------|---------:|-----------:|
| Cosmetics | M2          | AP       | 0.453709 | 0.0107437  |
| Cosmetics | M2          | Brier    | 0.226429 | 0.0128784  |
| Cosmetics | M2          | MCC      | 0.227468 | 0.0125445  |
| Cosmetics | M3          | AP       | 0.45947  | 0.0160831  |
| Cosmetics | M3          | Brier    | 0.229512 | 0.0179993  |
| Cosmetics | M3          | MCC      | 0.269877 | 0.0132791  |
| Cosmetics | M4          | AP       | 0.460016 | 0.0170114  |
| Cosmetics | M4          | Brier    | 0.216065 | 0.006528   |
| Cosmetics | M4          | MCC      | 0.279026 | 0.0278992  |
| Coveo     | M2          | AP       | 0.551851 | 0.00288902 |
| Coveo     | M2          | Brier    | 0.161356 | 0.00257715 |
| Coveo     | M2          | MCC      | 0.486691 | 0.00736048 |
| Coveo     | M3          | AP       | 0.551042 | 0.00539754 |
| Coveo     | M3          | Brier    | 0.161662 | 0.0124131  |
| Coveo     | M3          | MCC      | 0.486525 | 0.00484647 |
| Coveo     | M4          | AP       | 0.549898 | 0.00610054 |
| Coveo     | M4          | Brier    | 0.161538 | 0.013501   |
| Coveo     | M4          | MCC      | 0.485351 | 0.00363421 |

Figure caption. Paired AP differences between Act+Rel+Dwell LTLf (M4) and Act+Rel LTLf (M3), conditional on the same raw, strong-summary, and nonlogical timing inputs. Bars show 95% percentile intervals from 1,500 shared cluster resamples (Cosmetics: user; Coveo: session). Neural points average paired differences over three optimization seeds using identical cluster resamples across seeds. Models, formulas, and thresholds are fixed. Only Coveo and Cosmetics are new experiments; the three-dataset historical dwell results are audited separately.
Neural paired differences across optimization seeds are additionally summarized by their mean and sample SD in neural_paired_mean_sd.csv. The cluster bootstrap keeps the three trained seed-specific models fixed; it does not estimate uncertainty over the distribution of optimization seeds.