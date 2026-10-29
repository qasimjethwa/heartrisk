"""HeartRisk training + export.

Trains logistic regression and random forest on the public Framingham-style
dataset, compares them honestly, and exports the selected model (logistic
regression) in a form the browser can reproduce exactly.

Leakage control:
  - All imputation / scaling lives inside sklearn Pipelines, fitted on the
    training split only. The test split is touched once, for evaluation.
  - Stratified 80/20 split (random_state=42) + stratified 5-fold CV on train.
  - `education` is dropped: it is not a health input in the UI and using it
    as a predictor would add a social variable with no clinical meaning here.

Outputs (paths are relative to the repository root):
  docs/model.json   — exported LR model + metrics + calibration + provenance
  METRICS.md        — human-readable comparison
  src/parity_cases.json — held-out + profile + edge cases for parity checking

Run:  python3 src/train.py
Requires: pandas, scikit-learn (see README).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "framingham.csv"

FEATURES = ["male", "age", "currentSmoker", "cigsPerDay", "BPMeds",
            "prevalentStroke", "prevalentHyp", "diabetes", "totChol",
            "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
TARGET = "TenYearCHD"

LABELS = {
    "male": "Sex (male = 1)", "age": "Age (years)",
    "currentSmoker": "Current smoker", "cigsPerDay": "Cigarettes per day",
    "BPMeds": "On BP medication", "prevalentStroke": "Prevalent stroke",
    "prevalentHyp": "Prevalent hypertension", "diabetes": "Diabetes",
    "totChol": "Total cholesterol (mg/dL)", "sysBP": "Systolic BP (mmHg)",
    "diaBP": "Diastolic BP (mmHg)", "BMI": "Body mass index",
    "heartRate": "Heart rate (bpm)", "glucose": "Glucose (mg/dL)",
}
# Reference value used by the UI for "top factors": training median for
# continuous features, 0 (absent) for binary features.
REFERENCE = {"male": 0, "currentSmoker": 0, "BPMeds": 0, "prevalentStroke": 0,
             "prevalentHyp": 0, "diabetes": 0}


def evaluate(model, X_test, y_test):
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, prob)), 4),
        "f1_at_0.5": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "average_precision": round(float(average_precision_score(y_test, prob)), 4),
        "brier_score": round(float(brier_score_loss(y_test, prob)), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "n_test": int(len(y_test)),
    }


def main():
    df = pd.read_csv(DATA)
    assert list(df.columns)[:16] == ["male", "age", "education", "currentSmoker",
        "cigsPerDay", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
        "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose", "TenYearCHD"], \
        "Dataset columns changed — re-check provenance before proceeding."
    X, y = df[FEATURES], df[TARGET].astype(int)
    missing = {c: int(X[c].isna().sum()) for c in FEATURES}
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    lr = Pipeline([("imputer", SimpleImputer(strategy="median")),
                   ("scaler", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=1000, C=1.0))])
    rf = Pipeline([("imputer", SimpleImputer(strategy="median")),
                   ("clf", RandomForestClassifier(n_estimators=300,
                        min_samples_leaf=2, random_state=42, n_jobs=-1))])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, model in [("logistic_regression", lr), ("random_forest", rf)]:
        model.fit(X_train, y_train)
        m = evaluate(model, X_test, y_test)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        m["cv_roc_auc_mean"] = round(float(scores.mean()), 4)
        m["cv_roc_auc_std"] = round(float(scores.std()), 4)
        m["cv_roc_auc_folds"] = [round(float(s), 4) for s in scores]
        results[name] = m
        print(name, m)

    # Calibration for the selected model, on the test split (reported, and
    # used to decide whether a calibrated export is worth its complexity).
    lr_prob = lr.predict_proba(X_test)[:, 1]
    frac, mean_pred = calibration_curve(y_test, lr_prob, n_bins=10, strategy="uniform")
    calibration = [{"mean_predicted": round(float(a), 4),
                    "fraction_positive": round(float(b), 4)}
                   for a, b in zip(mean_pred, frac)]

    # Selection: LR wins on test AUC (0.702 vs 0.646) and CV AUC, is exactly
    # exportable (coefficients + scaler), and is explainable. Sigmoid
    # calibration was tried in prototyping and improved Brier by only
    # ~0.0004, so the uncalibrated LR is shipped and the calibration curve
    # is published instead of hiding the miscalibration behind an ensemble.
    imp = lr.named_steps["imputer"]
    sc = lr.named_steps["scaler"]
    clf = lr.named_steps["clf"]
    medians = {f: float(v) for f, v in zip(FEATURES, imp.statistics_)}
    reference = {f: REFERENCE.get(f, medians[f]) for f in FEATURES}
    ranges = {f: [float(X[f].min()), float(X[f].max())] for f in FEATURES}

    model = {
        "model": "logistic_regression",
        "features": FEATURES,
        "labels": LABELS,
        "reference": reference,
        "medians": medians,
        "scaler_mean": [float(v) for v in sc.mean_],
        "scaler_scale": [float(v) for v in sc.scale_],
        "coefficients": [float(v) for v in clf.coef_[0]],
        "intercept": float(clf.intercept_[0]),
        "metrics": results,
        "calibration_bins": calibration,
        "selection_reason": (
            "Logistic regression had the higher test ROC-AUC and CV ROC-AUC, "
            "exports exactly as coefficients + scaler (faithful browser parity), "
            "and is directly explainable. Random forest was lower on test AUC "
            "and would export as hundreds of trees. Sigmoid calibration improved "
            "Brier by only ~0.0004 in prototyping, so it was not shipped; the "
            "calibration curve is published instead."),
        "dataset": {
            "rows": int(len(df)),
            "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
            "prevalence_overall": round(float(y.mean()), 4),
            "prevalence_train": round(float(y_train.mean()), 4),
            "prevalence_test": round(float(y_test.mean()), 4),
            "missing_per_feature": missing,
            "training_ranges": ranges,
            "dropped_feature": "education (not a health input; excluded deliberately)",
            "source_file": "data/framingham.csv",
        },
        "display_bands": {"low_max": 0.05, "moderate_max": 0.20,
                          "note": "Display bands only, not clinical thresholds."},
    }
    (ROOT / "docs" / "model.json").write_text(json.dumps(model, indent=2))
    print("Wrote docs/model.json")

    # Parity cases: 30 held-out rows + illustrative profiles + edge cases.
    cases = []
    for _, row in X_test.sample(30, random_state=7).iterrows():
        cases.append({f: (None if pd.isna(row[f]) else float(row[f])) for f in FEATURES})
    profiles = {
        "a_52_smoker": {"male": 1, "age": 52, "currentSmoker": 1, "cigsPerDay": 20,
            "BPMeds": 0, "prevalentStroke": 0, "prevalentHyp": 1, "diabetes": 0,
            "totChol": 240, "sysBP": 138, "diaBP": 85, "BMI": 27.5, "heartRate": 75, "glucose": 85},
        "b_48_nonsmoker": {"male": 0, "age": 48, "currentSmoker": 0, "cigsPerDay": 0,
            "BPMeds": 0, "prevalentStroke": 0, "prevalentHyp": 0, "diabetes": 0,
            "totChol": 200, "sysBP": 118, "diaBP": 76, "BMI": 24.0, "heartRate": 70, "glucose": 80},
        "c_61_higher": {"male": 1, "age": 61, "currentSmoker": 0, "cigsPerDay": 0,
            "BPMeds": 1, "prevalentStroke": 0, "prevalentHyp": 1, "diabetes": 1,
            "totChol": 260, "sysBP": 150, "diaBP": 92, "BMI": 30.0, "heartRate": 80, "glucose": 110},
    }
    for name, prof in profiles.items():
        cases.append({"_profile": name, **prof})
    # Edge cases: extremes, missing (null -> median imputation), non-smoker
    # with cigarettes set (UI zeroes these, but the model path must not break).
    cases.append({"_profile": "edge_min", **{f: ranges[f][0] for f in FEATURES}})
    cases.append({"_profile": "edge_max", **{f: ranges[f][1] for f in FEATURES}})
    cases.append({"_profile": "edge_all_missing", **{f: None for f in FEATURES}})
    cases.append({"_profile": "edge_nonsmoker_with_cigs",
                  **profiles["b_48_nonsmoker"], "cigsPerDay": 20})
    (ROOT / "src" / "parity_cases.json").write_text(json.dumps(cases, indent=2))
    # Python-side expected probabilities for those cases (sklearn, not manual).
    frame = pd.DataFrame([{f: c.get(f) for f in FEATURES} for c in cases])
    probs = lr.predict_proba(frame)[:, 1]
    expected = [{"case": c.get("_profile", f"heldout_{i}"),
                 "python_probability": float(p)} for i, (c, p) in enumerate(zip(cases, probs))]
    (ROOT / "src" / "parity_expected.json").write_text(json.dumps(expected, indent=2))
    print("Wrote parity cases + expected probabilities")

    lr_m, rf_m = results["logistic_regression"], results["random_forest"]
    (ROOT / "METRICS.md").write_text(f"""# HeartRisk — model metrics

Selected model: **logistic regression** (see `docs/model.json` for the full export).

## Held-out test set (n = {lr_m['n_test']}, stratified, random_state = 42)

| Model | Accuracy | ROC-AUC | F1 @ 0.5 | Avg precision | Brier |
|---|---|---|---|---|---|
| Logistic regression | {lr_m['accuracy']} | {lr_m['roc_auc']} | {lr_m['f1_at_0.5']} | {lr_m['average_precision']} | {lr_m['brier_score']} |
| Random forest | {rf_m['accuracy']} | {rf_m['roc_auc']} | {rf_m['f1_at_0.5']} | {rf_m['average_precision']} | {rf_m['brier_score']} |

Cross-validated ROC-AUC on training data (5-fold, stratified):
logistic regression {lr_m['cv_roc_auc_mean']} ± {lr_m['cv_roc_auc_std']};
random forest {rf_m['cv_roc_auc_mean']} ± {rf_m['cv_roc_auc_std']}.

Confusion matrix, logistic regression (rows = actual, cols = predicted):
`{lr_m['confusion_matrix']}` — at the 0.5 threshold the model rarely predicts
the positive class, so F1 is low despite moderate AUC. This is the honest
consequence of a 15.2% prevalence outcome and a 0.5 threshold; the shipped
product therefore shows probabilities and bands, never a bare yes/no label.

## Why logistic regression was selected

{model['selection_reason']}

## Calibration (test set, logistic regression)

Mean predicted vs fraction positive per bin: `{calibration}`.
Brier score: {lr_m['brier_score']}. The curve is published in full in
`docs/model.json` (`calibration_bins`).

## Limitations

See README.md and docs/methodology.html. In short: an old, narrow cohort;
15.2% prevalence; missing glucose in 388/4,240 rows (median-imputed from
train only); no causal interpretation; no clinical validation.
""")
    print("Wrote METRICS.md")


if __name__ == "__main__":
    main()
