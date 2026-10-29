# HeartRisk — Python ↔ browser parity (current, calibrated model)

- **Method:** `src/train.py` writes 43 parity cases to
  `src/parity_cases.json` (30 held-out test rows from the new 60/20/20
  split sampled with `random_state=7`, 3 illustrative profiles, 4 edge
  cases: training-range minimum, training-range maximum, all features
  missing, and a non-smoker profile with cigarettes/day set to 20, plus
  6 calibration cases: low/mid/high risk, quit smoking, BP −20, and
  cholesterol −40) and their shipped-model probabilities to
  `src/parity_expected.json`. The shipped probability is the sklearn
  logistic-regression probability passed through the sigmoid (Platt)
  calibration fitted on the calibration split only.
  `src/parity.js` loads the exact browser code (`docs/model-core.js`)
  and `docs/model.json` under Node and recomputes every case.
- **Tolerance:** 1e-4 absolute probability difference.
- **Result (run 2026-10-30, after the evaluation rework):** 43 cases,
  maximum absolute difference **8.327e-17** (worst case:
  cal_chol_down_40) — **PARITY PASS**.
  The difference is floating-point rounding only; the browser formula
  (median imputation → standard scaling → logistic sigmoid → sigmoid
  calibration) reproduces the Python pipeline exactly.
- **Re-run:** `node src/parity.js` from the repository root (exits
  non-zero on failure).
- **Edge-case behaviour verified by the same cases:** all-missing inputs
  fall back to training medians (no crash, no NaN); extreme inputs produce
  probabilities in [0, 1]; a non-smoker profile with cigarettes set does
  not break the model path (the UI separately zeroes cigarettes for
  non-smokers).
- **Superseded result:** the previous 37-case parity pass (~1e-16) was for
  the old uncalibrated 80/20 model and must not be quoted for the current
  calibrated model; the 43-case result above replaces it.
