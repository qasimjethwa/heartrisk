# HeartRisk — data provenance

## Dataset used
- **What:** Framingham-style cardiovascular cohort data, 4,240 rows × 16 columns.
  Target: `TenYearCHD` — coronary heart disease within a 10-year follow-up (1 = yes).
  Overall prevalence in the file: 644 / 4,240 = 15.2%.
- **Columns:** male, age, education, currentSmoker, cigsPerDay, BPMeds,
  prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI,
  heartRate, glucose, TenYearCHD.
- **File used for training:** `data/framingham.csv`
  (SHA-256: `2c0e57dc0361b420becf1facec0a054af06c420eae0ae2faf0fdc8591fadb018`).
  First data row verified against published examples (39-year-old male,
  cholesterol 195, systolic 106, TenYearCHD 0) — the file matches the
  widely circulated Kaggle Framingham copy.
- **Download source (verified via the GitHub API listing):**
  https://github.com/GauravPadawe/Framingham-Heart-Study
  — file `framingham.csv`, downloaded from
  https://raw.githubusercontent.com/GauravPadawe/Framingham-Heart-Study/master/framingham.csv
- **Upstream origin stated in that repository's README:**
  the Framingham Heart Study (NHLBI), via the Biolincc teaching dataset
  (https://biolincc.nhlbi.nih.gov/teaching/), and the Kaggle copy published
  for a machine-learning competition. The R `LSTbook` documentation describes
  the same 4,238/4,240-subject Kaggle-derived dataset.

## Licence — caveat, read before republishing
- The mirror repository contains **no licence file**, and the NHLBI
  Biolincc teaching data is described by its host as "download on request".
- A secondary source (an RMIT assignment write-up) describes the Kaggle
  copy as Creative Commons Attribution 1.0 Universal, but that claim could
  **not** be verified against a primary licence statement for this file.
- **Therefore this project does not assert a licence for the dataset.**
  The raw CSV is kept in the local build for reproducibility; before
  publishing the repository, the publisher should confirm the dataset's
  licence / redistribution terms, or remove `data/framingham.csv` from the
  public repo and keep only the trained-model export (`docs/model.json`),
  which contains no individual-level data (only medians, scaler statistics,
  coefficients and aggregate metrics).

## What was NOT done
- No synthetic data was substituted for real data, and no rows were invented.
- No individual in the dataset is identified anywhere in this project;
  example profiles in the demo are illustrative inventions, labelled as such.
- The dataset is a mid-20th-century US cohort (ages 32–70). It is not
  representative of a modern global population, and this project makes no
  claim that it is.

## Missing data (rows with missing values, out of 4,240)
glucose 388, education 105, BPMeds 53, cigsPerDay 29, BMI 19, heartRate 1.
Training used median imputation fitted on the training split only.
