# HeartRisk — complete evaluation report

> **Clinical-evidence warning (read first).** There is NOT yet enough
> evidence to trust these individual risk estimates for clinical decisions.
> What is shown here is a single held-out test evaluation on one historical
> cohort, with calibration fitted on one calibration split. Stronger
> calibration evidence and evaluation on an independent, relevant
> population are still required. This product is educational only, not a
> diagnosis, not medical advice, and not clinically validated.

## 1 · Held-out design (no test peeking for selection)
- Data: public Framingham-style `data/framingham.csv`, 4,240 rows,
  target `TenYearCHD`, overall prevalence 15.2%. Licence: unverified for the
  mirror (see DATA_PROVENANCE.md); no licence is asserted. The shipped
  `docs/model.json` contains no individual-level data.
- Split: stratified 60% train / 20% calibration / 20% test,
  `random_state=42` — train 2544 rows
  (prevalence 0.1517), calibration
  848 rows
  (prevalence 0.1521), test
  848 rows (prevalence 0.1521).
- Preprocessing inside Pipelines, fitted on train only: median imputation
  for all models; standard scaling for logistic regression only.
- Tuning: stratified 5-fold CV on TRAIN only, scoring ROC-AUC.
- Calibration: fitted on the CALIBRATION split only. Method chosen by
  5-fold cross-fitted evaluation within the calibration split (see §4),
  not by test results. Test was evaluated once, finally, for all
  candidates and variants below.
- Features (14): male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose. `education` dropped deliberately.
- Missing values (full data): {'male': 0, 'age': 0, 'currentSmoker': 0, 'cigsPerDay': 29, 'BPMeds': 53, 'prevalentStroke': 0, 'prevalentHyp': 0, 'diabetes': 0, 'totChol': 50, 'sysBP': 0, 'diaBP': 0, 'BMI': 19, 'heartRate': 1, 'glucose': 388}.
- XGBoost: not installed in this environment, so it was not compared and
  no heavy dependency was added silently. HistGradientBoostingClassifier
  (sklearn) is the gradient-boosting candidate instead.

## 2 · Tuning results (train CV only)
| Candidate | Best parameters | CV ROC-AUC |
|---|---|---|
| Regularized logistic regression | {'clf__C': 0.01, 'clf__class_weight': 'balanced'} | 0.7457 ± 0.0121 |
| Random forest | {'clf__max_depth': 6, 'clf__max_features': 'sqrt', 'clf__min_samples_leaf': 1, 'clf__n_estimators': 200} | 0.7378 ± 0.0049 |
| HistGradientBoosting | {'clf__l2_regularization': 0.1, 'clf__learning_rate': 0.05, 'clf__max_depth': 3, 'clf__max_iter': 200} | 0.7062 ± 0.0169 |
| Unregularized LR baseline (C=1e6) | (no tuning) | 0.742 ± 0.0156 |

Class weight: the tuned logistic regression selected
`class_weight='balanced'` with a very strong penalty (C=0.01). That
improved ranking (CV AUC) but badly inflated raw probabilities — see §4.
That trade-off is exactly why calibration, not AUC alone, decides whether
a probability can be shown to a user.

## 3 · Test-set results (n=848, untouched until final evaluation)

### Sigmoid-calibrated
| Model | Accuracy | ROC-AUC | PR-AUC | Precision | Recall | Specificity | F1@0.5 | Confusion [[TN,FP],[FN,TP]] | Brier | Log loss | Cal. intercept | Cal. slope |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lr_regularized | 0.8514 | 0.7041 | 0.3021 | 0.6364 | 0.0543 | 0.9944 | 0.1 | [[715, 4], [122, 7]] | 0.1205 | 0.3934 | -0.1093 | 0.9416 |
| random_forest | 0.8467 | 0.6777 | 0.2808 | 0.4 | 0.0155 | 0.9958 | 0.0299 | [[716, 3], [127, 2]] | 0.1221 | 0.4004 | -0.1601 | 0.9167 |
| hist_gb | 0.8491 | 0.6809 | 0.2784 | 1.0 | 0.0078 | 1.0 | 0.0154 | [[719, 0], [128, 1]] | 0.1222 | 0.4 | 0.0072 | 1.0271 |
| lr_unregularized_baseline | 0.8491 | 0.7047 | 0.3001 | 0.5714 | 0.031 | 0.9958 | 0.0588 | [[716, 3], [125, 4]] | 0.1202 | 0.3927 | -0.065 | 0.9693 |

### Uncalibrated
| Model | Accuracy | ROC-AUC | PR-AUC | Precision | Recall | Specificity | F1@0.5 | Confusion [[TN,FP],[FN,TP]] | Brier | Log loss | Cal. intercept | Cal. slope |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lr_regularized | 0.6745 | 0.7041 | 0.3021 | 0.2508 | 0.5736 | 0.6926 | 0.3491 | [[498, 221], [55, 74]] | 0.207 | 0.6024 | -1.633 | 0.8223 |
| random_forest | 0.8467 | 0.6777 | 0.2808 | 0.4 | 0.0155 | 0.9958 | 0.0299 | [[716, 3], [127, 2]] | 0.1221 | 0.4007 | -0.1672 | 0.8698 |
| hist_gb | 0.8396 | 0.6809 | 0.2784 | 0.36 | 0.0698 | 0.9777 | 0.1169 | [[703, 16], [120, 9]] | 0.1249 | 0.4088 | -0.5535 | 0.6141 |
| lr_unregularized_baseline | 0.8467 | 0.7047 | 0.3001 | 0.4762 | 0.0775 | 0.9847 | 0.1333 | [[708, 11], [119, 10]] | 0.1215 | 0.3963 | -0.3501 | 0.7441 |

### Isotonic-calibrated
| Model | Accuracy | ROC-AUC | PR-AUC | Precision | Recall | Specificity | F1@0.5 | Confusion [[TN,FP],[FN,TP]] | Brier | Log loss | Cal. intercept | Cal. slope |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lr_regularized | 0.8455 | 0.7038 | 0.2796 | 0.45 | 0.0698 | 0.9847 | 0.1208 | [[708, 11], [120, 9]] | 0.1224 | 0.5918 | -1.3646 | 0.2019 |
| random_forest | 0.8479 | 0.6707 | 0.2533 | 0.0 | 0.0 | 1.0 | 0.0 | [[719, 0], [129, 0]] | 0.123 | 0.4027 | -0.4267 | 0.7411 |
| hist_gb | 0.8491 | 0.6655 | 0.2441 | 1.0 | 0.0078 | 1.0 | 0.0154 | [[719, 0], [128, 1]] | 0.1244 | 0.4057 | -0.4049 | 0.7593 |
| lr_unregularized_baseline | 0.8479 | 0.7015 | 0.2799 | 0.5 | 0.0543 | 0.9903 | 0.0979 | [[712, 7], [122, 7]] | 0.1233 | 0.5566 | -1.3308 | 0.212 |

Notes on reading this table:
- ROC-AUC is ranking only (0.704 for the shipped model — modest). PR-AUC
  (average precision) is the minority-class view: 0.302 for the shipped
  model, against a 0.152 prevalence baseline.
- At the default 0.5 threshold the shipped calibrated model still catches
  only 5.4% of positives (precision
  63.6%, confusion [[715, 4], [122, 7]]).
  That is why the UI never shows a bare yes/no verdict: a 0.5 cut on a
  15% prevalence outcome is a poor summary, and the probability itself is
  the product.
- Accuracy (~85%) is dominated by the majority class and must not be
  quoted alone.

## 4 · Calibration (the probability-trust question)

Method for intercept/slope (reporting only, fitted on test): logistic
regression of the observed outcome on logit(predicted probability).
Perfect calibration is intercept 0, slope 1. For isotonic (step-function)
outputs this slope is unstable — a step function has few distinct
probabilities, so a logit-slope fit on it is not a reliable summary; it
is reported for completeness and never used alone to choose isotonic.

Cross-fitted calibration on the calibration split (honest, out-of-fold):
| Candidate | Uncalibrated Brier | Sigmoid Brier | Isotonic Brier | Uncal. log loss | Sigmoid log loss | Isotonic log loss |
|---|---|---|---|---|---|---|
| LR regularized | 0.2059 | 0.1194 | 0.1193 | 0.598 | 0.394 | 0.4334 |
| Random forest | 0.1217 | 0.1218 | 0.1224 | 0.399 | 0.3997 | 0.4017 |
| HistGradientBoosting | 0.1231 | 0.1228 | 0.1235 | 0.4125 | 0.4055 | 0.4456 |
| LR unregularized | 0.1198 | 0.1196 | 0.1206 | 0.397 | 0.395 | 0.5151 |

Decision: sigmoid was shipped for the tuned LR. Its cross-fitted Brier
(0.1194) is
essentially tied with isotonic
(0.1193),
but its cross-fitted log loss is clearly better
(0.394 vs
0.4334),
it is a smooth two-parameter map that exports exactly to the browser, and
its test intercept/slope (-0.1093 /
0.9416) are close to 0/1. The raw tuned-LR
probabilities must NOT be used: uncalibrated test Brier
0.207 with
intercept -1.633
means they systematically overstate risk.

Plain-language meaning: if 100 people are given a predicted risk of 20%,
a calibrated model means about 20 of them should experience the outcome
over 10 years, on average, in an appropriate population. It will not be
exactly 20 every time. The shipped model's test calibration bins
(decile bins, n≈85 each, no small-n bins) are in `docs/model.json` and
charted in the demo. The top bin (mean predicted
0.3899, observed
0.3412, n=85)
still has sampling noise of roughly ±5 percentage points, and the whole
curve is estimated on one historical cohort — it is not proof the
probabilities hold for any other population.

Shipped calibration parameters (fitted on calibration split only):
a=-1.6183, b=0.8737,
formula p_cal = sigmoid(a + b·logit(p_raw)). Implemented identically in
`docs/model-core.js` and verified by parity (see PARITY.md).

## 5 · Selection reason
Regularized logistic regression won on train-CV ROC-AUC (0.7457 vs random forest 0.7378 and HistGradientBoosting 0.7062; unregularized baseline 0.742), with best parameters {'clf__C': 0.01, 'clf__class_weight': 'balanced'}. Its raw probabilities were badly miscalibrated because class_weight='balanced' inflates them (uncalibrated test Brier 0.207, intercept -1.633); sigmoid (Platt) calibration fitted on the calibration split only (a=-1.6183, b=0.8737) restores them (cross-fitted calibration Brier 0.1194 vs uncalibrated 0.2059; test Brier 0.1205, intercept -0.1093, slope 0.9416). Isotonic was tied on cross-fitted Brier but worse on cross-fitted log loss (0.4334 vs 0.394) and unstable on test, so sigmoid was shipped. Random forest / HistGradientBoosting were not shipped: lower CV AUC and no faithful lightweight browser export.

## 6 · Limitations that remain, even after this fix
- Single historical cohort (mid-20th-century Framingham, ages 32–70);
  no external or independent-population validation was done here.
- Modest discrimination (test ROC-AUC 0.704, PR-AUC 0.302) and poor
  recall at 0.5 — the model ranks better than it classifies.
- Calibration is estimated on 848 test cases; high-risk bins are noisy
  and the curve may not transport to other populations, ages, eras or
  healthcare settings.
- No former-smoker category, binary sex coding, no family history,
  diet, activity, genetics or medication detail in the source data.
- Scenario changes are model associations, not causal effects, and do
  not account for factors moving together in real life.
- No clinical validation, regulatory review or prospective testing.
