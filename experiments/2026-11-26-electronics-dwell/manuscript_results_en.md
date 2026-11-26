# Three-dataset results and manuscript material

Across all three datasets, the six primary M4–M3 AP intervals (LR and three-seed neural means) included zero. The extension therefore provides no consistent evidence that adding the dwell vocabulary improves AP; it does not establish equivalence. Electronics yielded −0.03988 [−0.12317, +0.02018] for LR and −0.00125 [−0.00585, +0.00359] for the neural mean.
Secondary outcomes were mixed or adverse. For Electronics, LR M2–M0 reduced MCC by 0.16438 (interval −0.30918 to −0.00570), while LR M4–M2 increased Brier by 0.01445 (0.00231 to 0.02828). Neural M4–M3 reduced mean MCC by 0.11741 (−0.17797 to −0.04581), concentrated in seed42: the three paired differences were −0.32967, −0.02255, and 0. Validation-selected thresholds are especially uncertain with only eight validation positives.
Neural M4–M2 improved mean Brier by 0.01602 (−0.02227 to −0.01030), but its seed differences were +0.01486, −0.01806, and −0.04487. M3–M2 had a slightly larger mean improvement (−0.01660), so this benefit cannot be attributed specifically to the dwell vocabulary. These conditional, unadjusted intervals should be interpreted alongside seed variation and the primary AP outcome.
Seven of the ten Electronics M4 formulas contained known-gap buckets, none contained the unknown marker, and nine contained temporal operators; the remaining formula was the atomic proposition dl. No formula exactly matched the enumerated univariate dwell-count thresholds on training rows, which does not rule out other forms of redundancy. Candidate mining took approximately 436 and 513 seconds; the four selection procedures required approximately 305 seconds and 6,260 candidate refits. These local timings do not establish end-to-end acceleration.

The completed Coveo and Cosmetics experiments were retained unchanged. Electronics was added at the user’s request after those results were available; this is a sequential exploratory extension, not an originally prespecified third dataset. All new Electronics models, formulas and thresholds were frozen before its test evaluation. Each dataset uses event-20 observation, a recent ten-event raw window, and first purchase within the next ten recognized events.

For Electronics, LR M2-M0 yielded an AP difference of -0.02054 [-0.15995, +0.11499].
For Electronics, LR M4-M3 yielded an AP difference of -0.03988 [-0.12317, +0.02018].
For Electronics, LR M4-M2 yielded an AP difference of -0.03308 [-0.09547, +0.00809].
For Electronics, LR M4-M5 yielded an AP difference of +0.02098 [-0.03351, +0.08139].
For Electronics, NN_mean M4-M3 yielded an AP difference of -0.00125 [-0.00585, +0.00359].
For Electronics, NN_mean M4-M2 yielded an AP difference of +0.01084 [-0.05288, +0.07017].
For Electronics, NN_mean M3-M2 yielded an AP difference of +0.01209 [-0.04998, +0.07043].

Electronics contains 558 training sessions (95 positives), 76 validation sessions (8 positives), and 171 test sessions (27 positives; 170 users). The small validation-positive count limits model-selection stability and the strength of generalization claims.

Electronics secondary contrasts (higher MCC and lower Brier are preferable):

| model   | contrast   | metric   |        delta |     lower_95 |    upper_95 |
|:--------|:-----------|:---------|-------------:|-------------:|------------:|
| LR      | M4-M3      | MCC      | -0.181729    | -0.349885    |  0.0101719  |
| LR      | M4-M3      | Brier    |  0.0108211   | -0.00527371  |  0.0274915  |
| LR      | M2-M0      | MCC      | -0.164379    | -0.309177    | -0.00569851 |
| LR      | M2-M0      | Brier    |  0.0285363   | -0.000123082 |  0.0558108  |
| LR      | M4-M2      | MCC      |  0.051031    | -0.137635    |  0.249358   |
| LR      | M4-M2      | Brier    |  0.0144534   |  0.00231358  |  0.028283   |
| LR      | M4-M5      | MCC      | -0.0779585   | -0.184292    |  0          |
| LR      | M4-M5      | Brier    | -0.00369298  | -0.0244836   |  0.0152551  |
| NN_mean | M4-M3      | MCC      | -0.117409    | -0.177974    | -0.0458106  |
| NN_mean | M4-M3      | Brier    |  0.000581432 | -4.63221e-05 |  0.00121263 |
| NN_mean | M4-M2      | MCC      | -0.0966969   | -0.232919    |  0.0485     |
| NN_mean | M4-M2      | Brier    | -0.0160229   | -0.0222733   | -0.0103035  |
| NN_mean | M3-M2      | MCC      |  0.020712    | -0.0841108   |  0.120514   |
| NN_mean | M3-M2      | Brier    | -0.0166043   | -0.0230252   | -0.0107755  |

Per-seed Electronics differences:

| contrast   |   seed | metric   |        delta |     lower_95 |     upper_95 |
|:-----------|-------:|:---------|-------------:|-------------:|-------------:|
| M4-M3      |     42 | MCC      | -0.329674    | -0.495044    | -0.124523    |
| M4-M3      |     42 | AP       | -0.00170274  | -0.00622748  |  0.00343923  |
| M4-M3      |     42 | Brier    |  0.00050608  |  6.35475e-05 |  0.000926167 |
| M4-M3      |     43 | MCC      | -0.022553    | -0.085114    |  0.0126142   |
| M4-M3      |     43 | AP       |  0.000367192 | -0.00232626  |  0.00389989  |
| M4-M3      |     43 | Brier    |  0.000631324 |  0.000258864 |  0.00099331  |
| M4-M3      |     44 | MCC      |  0           |  0           |  0           |
| M4-M3      |     44 | AP       | -0.00241605  | -0.0134463   |  0.00939076  |
| M4-M3      |     44 | Brier    |  0.000606892 | -0.000840862 |  0.00213446  |
| M4-M2      |     42 | MCC      | -0.314742    | -0.47316     | -0.115615    |
| M4-M2      |     42 | AP       | -0.00595445  | -0.0825151   |  0.0686195   |
| M4-M2      |     42 | Brier    |  0.014858    |  0.00344335  |  0.0262272   |
| M4-M2      |     43 | MCC      |  0.0272648   | -0.0293762   |  0.10258     |
| M4-M2      |     43 | AP       |  0.00386744  | -0.087091    |  0.0865305   |
| M4-M2      |     43 | Brier    | -0.018056    | -0.0273733   | -0.0088643   |
| M4-M2      |     44 | MCC      | -0.00261329  | -0.28976     |  0.272359    |
| M4-M2      |     44 | AP       |  0.0346178   | -0.120063    |  0.153696    |
| M4-M2      |     44 | Brier    | -0.0448707   | -0.0684984   | -0.0212592   |
| M3-M2      |     42 | MCC      |  0.0149315   | -0.0706834   |  0.0998758   |
| M3-M2      |     42 | AP       | -0.00425172  | -0.0803102   |  0.0689487   |
| M3-M2      |     42 | Brier    |  0.0143519   |  0.00309707  |  0.0256199   |
| M3-M2      |     43 | MCC      |  0.0498179   | -0.0265834   |  0.148541    |
| M3-M2      |     43 | AP       |  0.00350024  | -0.0872378   |  0.0835239   |
| M3-M2      |     43 | Brier    | -0.0186873   | -0.0280527   | -0.00951557  |
| M3-M2      |     44 | MCC      | -0.00261329  | -0.28976     |  0.272359    |
| M3-M2      |     44 | AP       |  0.0370338   | -0.115095    |  0.155614    |
| M3-M2      |     44 | Brier    | -0.0454776   | -0.0686954   | -0.0223715   |

Of 10 selected Electronics M4 formulas, 7 referenced known-duration buckets and 0 referenced the unknown-gap marker; 9 contained temporal syntax. Electronics p2/p3 encode cumulative distinct-product counts. Count-threshold agreement is a limited training diagnostic, not proof of irreducible temporal information.
Sessions with multiple user IDs were excluded before any fit, with counts reported in cohort_audit.csv. Electronics otherwise retains the audited historical session split; eligible-cohort user overlaps are {'train_val': 6, 'train_test': 12, 'val_test': 0}. User-cluster bootstrap does not remove cross-split user overlap. These results therefore do not measure unseen-user generalization. Outgoing gaps are formed after truncation, the terminal gap is unknown, genuine zero gaps remain distinguishable, and timing transformations use training data only.
All comparisons are exploratory. Intervals condition on fitted models, selected formulas and validation thresholds; they omit mining and optimization-seed uncertainty and are not multiplicity-adjusted. Formulas selected by LR are transferred unchanged to neural models. No pooled cross-dataset significance claim is made. Historical session-purchase results are background, not paired controls for the new landmark task.

| dataset     | condition   |       AP |    Brier |      MCC |
|:------------|:------------|---------:|---------:|---------:|
| Cosmetics   | M0          | 0.404481 | 0.207781 | 0.182219 |
| Cosmetics   | M1          | 0.404974 | 0.207691 | 0.172627 |
| Cosmetics   | M2          | 0.409534 | 0.210913 | 0.213567 |
| Cosmetics   | M3          | 0.409709 | 0.210732 | 0.202495 |
| Cosmetics   | M4          | 0.410786 | 0.21027  | 0.219662 |
| Cosmetics   | M5          | 0.40974  | 0.211012 | 0.204988 |
| Coveo       | M0          | 0.505594 | 0.127658 | 0.468111 |
| Coveo       | M1          | 0.509207 | 0.127164 | 0.467374 |
| Coveo       | M2          | 0.522178 | 0.126446 | 0.464535 |
| Coveo       | M3          | 0.523457 | 0.126201 | 0.46658  |
| Coveo       | M4          | 0.524391 | 0.126271 | 0.460127 |
| Coveo       | M5          | 0.524442 | 0.125902 | 0.466359 |
| Electronics | M0          | 0.324059 | 0.133651 | 0.21541  |
| Electronics | M1          | 0.377275 | 0.133415 | 0.220973 |
| Electronics | M2          | 0.303516 | 0.162187 | 0.051031 |
| Electronics | M3          | 0.310316 | 0.165819 | 0.283791 |
| Electronics | M4          | 0.270437 | 0.17664  | 0.102062 |
| Electronics | M5          | 0.249452 | 0.180333 | 0.180021 |

| dataset     | condition   | AP                | Brier             | MCC               |
|:------------|:------------|:------------------|:------------------|:------------------|
| Cosmetics   | M2          | 0.45371 (0.01074) | 0.22643 (0.01288) | 0.22747 (0.01254) |
| Cosmetics   | M3          | 0.45947 (0.01608) | 0.22951 (0.01800) | 0.26988 (0.01328) |
| Cosmetics   | M4          | 0.46002 (0.01701) | 0.21606 (0.00653) | 0.27903 (0.02790) |
| Coveo       | M2          | 0.55185 (0.00289) | 0.16136 (0.00258) | 0.48669 (0.00736) |
| Coveo       | M3          | 0.55104 (0.00540) | 0.16166 (0.01241) | 0.48653 (0.00485) |
| Coveo       | M4          | 0.54990 (0.00610) | 0.16154 (0.01350) | 0.48535 (0.00363) |
| Electronics | M2          | 0.32013 (0.02137) | 0.21211 (0.02574) | 0.21238 (0.10733) |
| Electronics | M3          | 0.33222 (0.00469) | 0.19551 (0.01191) | 0.23310 (0.11895) |
| Electronics | M4          | 0.33097 (0.00577) | 0.19609 (0.01191) | 0.11569 (0.11850) |

Figure caption. Paired M4−M3 AP differences on three datasets at t=20, m=10 and H=10. Bars are 95% percentile intervals from 1,500 shared cluster bootstrap resamples, by user for Electronics and Cosmetics and by session for Coveo. Neural points average paired differences across seeds42–44 using identical cluster resamples, keeping each model fixed. Electronics is a subsequent exploratory extension; Coveo and Cosmetics retain the previously completed results.