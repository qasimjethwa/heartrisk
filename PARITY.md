# HeartRisk — Python ↔ browser parity

- **Method:** `src/train.py` writes 37 parity cases to
  `src/parity_cases.json` (30 held-out test rows sampled with
  `random_state=7`, 3 illustrative profiles, 4 edge cases: training-range
  minimum, training-range maximum, all features missing, and a non-smoker
  profile with cigarettes/day set to 20) and their sklearn
  `predict_proba` values to `src/parity_expected.json`.
  `src/parity.js` loads the exact browser code (`docs/model-core.js`) and
  `docs/model.json` under Node and recomputes every case.
- **Tolerance:** 1e-4 absolute probability difference.
- **Result (run 2026-10-30):** 37 cases, maximum absolute difference
  **1.110e-16** (worst case: heldout_8) — **PARITY PASS**.
  The difference is floating-point rounding only; the browser formula
  (median imputation → standard scaling → logistic sigmoid) reproduces
  the Python pipeline exactly.
- **Re-run:** `node src/parity.js` from the repository root (exits non-zero
  on failure).
- **Edge-case behaviour verified by the same cases:** all-missing inputs
  fall back to training medians (no crash, no NaN); extreme inputs produce
  probabilities in [0, 1]; a non-smoker profile with cigarettes set does not
  break the model path (the UI separately zeroes cigarettes for non-smokers).
