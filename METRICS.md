# HeartRisk — model metrics (current evaluation)

> **Supersedes the previous 80/20 evaluation.** Old numbers (test ROC-AUC
> 0.702, F1 0.096, Brier 0.1213, confusion [[709,10],[122,7]]) are replaced
> by the 60/20/20 split below and must not be quoted alongside it.

Split: stratified 60% train / 20% calibration / 20% test, random_state 42 — train
2544, calibration 848,
test 848; prevalence train
0.1517, calibration
0.1521, test 0.1521.

Shipped model: regularized logistic regression (C and class_weight tuned by
5-fold CV on train only) + sigmoid (Platt) calibration fitted on the
calibration split only. See EVALUATION.md for the complete report.

## Test set, sigmoid-calibrated variants
| Model | Accuracy | ROC-AUC | PR-AUC | Precision | Recall | Specificity | F1@0.5 | Confusion [[TN,FP],[FN,TP]] | Brier | Log loss | Cal. intercept | Cal. slope |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lr_regularized | 0.8514 | 0.7041 | 0.3021 | 0.6364 | 0.0543 | 0.9944 | 0.1 | [[715, 4], [122, 7]] | 0.1205 | 0.3934 | -0.1093 | 0.9416 |
| random_forest | 0.8467 | 0.6777 | 0.2808 | 0.4 | 0.0155 | 0.9958 | 0.0299 | [[716, 3], [127, 2]] | 0.1221 | 0.4004 | -0.1601 | 0.9167 |
| hist_gb | 0.8491 | 0.6809 | 0.2784 | 1.0 | 0.0078 | 1.0 | 0.0154 | [[719, 0], [128, 1]] | 0.1222 | 0.4 | 0.0072 | 1.0271 |
| lr_unregularized_baseline | 0.8491 | 0.7047 | 0.3001 | 0.5714 | 0.031 | 0.9958 | 0.0588 | [[716, 3], [125, 4]] | 0.1202 | 0.3927 | -0.065 | 0.9693 |

## Test set, uncalibrated variants (for comparison)
| Model | Accuracy | ROC-AUC | PR-AUC | Precision | Recall | Specificity | F1@0.5 | Confusion [[TN,FP],[FN,TP]] | Brier | Log loss | Cal. intercept | Cal. slope |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lr_regularized | 0.6745 | 0.7041 | 0.3021 | 0.2508 | 0.5736 | 0.6926 | 0.3491 | [[498, 221], [55, 74]] | 0.207 | 0.6024 | -1.633 | 0.8223 |
| random_forest | 0.8467 | 0.6777 | 0.2808 | 0.4 | 0.0155 | 0.9958 | 0.0299 | [[716, 3], [127, 2]] | 0.1221 | 0.4007 | -0.1672 | 0.8698 |
| hist_gb | 0.8396 | 0.6809 | 0.2784 | 0.36 | 0.0698 | 0.9777 | 0.1169 | [[703, 16], [120, 9]] | 0.1249 | 0.4088 | -0.5535 | 0.6141 |
| lr_unregularized_baseline | 0.8467 | 0.7047 | 0.3001 | 0.4762 | 0.0775 | 0.9847 | 0.1333 | [[708, 11], [119, 10]] | 0.1215 | 0.3963 | -0.3501 | 0.7441 |

## Why this model, in one paragraph
Regularized logistic regression won on train-CV ROC-AUC (0.7457 vs random forest 0.7378 and HistGradientBoosting 0.7062; unregularized baseline 0.742), with best parameters {'clf__C': 0.01, 'clf__class_weight': 'balanced'}. Its raw probabilities were badly miscalibrated because class_weight='balanced' inflates them (uncalibrated test Brier 0.207, intercept -1.633); sigmoid (Platt) calibration fitted on the calibration split only (a=-1.6183, b=0.8737) restores them (cross-fitted calibration Brier 0.1194 vs uncalibrated 0.2059; test Brier 0.1205, intercept -0.1093, slope 0.9416). Isotonic was tied on cross-fitted Brier but worse on cross-fitted log loss (0.4334 vs 0.394) and unstable on test, so sigmoid was shipped. Random forest / HistGradientBoosting were not shipped: lower CV AUC and no faithful lightweight browser export.

## Calibration — plain language
If 100 people are given a predicted risk of 20%, a calibrated model means
about 20 of them should experience the outcome over 10 years, on average,
in an appropriate population. It will not be exactly 20 every time; what
matters is agreement across the whole range of risks. The shipped model's
test calibration bins are in `docs/model.json` (`calibration_bins`) and
charted in the demo's Science section and Methodology page.
