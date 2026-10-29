# HeartRisk — model metrics

Selected model: **logistic regression** (see `docs/model.json` for the full export).

## Held-out test set (n = 848, stratified, random_state = 42)

| Model | Accuracy | ROC-AUC | F1 @ 0.5 | Avg precision | Brier |
|---|---|---|---|---|---|
| Logistic regression | 0.8443 | 0.7021 | 0.0959 | 0.2989 | 0.1213 |
| Random forest | 0.8443 | 0.6458 | 0.0704 | 0.265 | 0.1264 |

Cross-validated ROC-AUC on training data (5-fold, stratified):
logistic regression 0.732 ± 0.0307;
random forest 0.7194 ± 0.0293.

Confusion matrix, logistic regression (rows = actual, cols = predicted):
`[[709, 10], [122, 7]]` — at the 0.5 threshold the model rarely predicts
the positive class, so F1 is low despite moderate AUC. This is the honest
consequence of a 15.2% prevalence outcome and a 0.5 threshold; the shipped
product therefore shows probabilities and bands, never a bare yes/no label.

## Why logistic regression was selected

Logistic regression had the higher test ROC-AUC and CV ROC-AUC, exports exactly as coefficients + scaler (faithful browser parity), and is directly explainable. Random forest was lower on test AUC and would export as hundreds of trees. Sigmoid calibration improved Brier by only ~0.0004 in prototyping, so it was not shipped; the calibration curve is published instead.

## Calibration (test set, logistic regression)

Mean predicted vs fraction positive per bin: `[{'mean_predicted': 0.0589, 'fraction_positive': 0.0672}, {'mean_predicted': 0.1437, 'fraction_positive': 0.1765}, {'mean_predicted': 0.2453, 'fraction_positive': 0.2264}, {'mean_predicted': 0.3419, 'fraction_positive': 0.3279}, {'mean_predicted': 0.4451, 'fraction_positive': 0.3182}, {'mean_predicted': 0.5538, 'fraction_positive': 0.2727}, {'mean_predicted': 0.6361, 'fraction_positive': 0.6}, {'mean_predicted': 0.7773, 'fraction_positive': 1.0}]`.
Brier score: 0.1213. The curve is published in full in
`docs/model.json` (`calibration_bins`).

## Limitations

See README.md and docs/methodology.html. In short: an old, narrow cohort;
15.2% prevalence; missing glucose in 388/4,240 rows (median-imputed from
train only); no causal interpretation; no clinical validation.
