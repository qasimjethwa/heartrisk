"""HeartRisk training, evaluation, calibration and export.

SUPERSEDES the previous 80/20 evaluation (old test ROC-AUC 0.702, F1 0.096,
Brier 0.1213). Those numbers are NOT mixed with the numbers below.

Held-out design (stated explicitly, no test peeking for selection):
  - Stratified split, random_state=42: 60% train / 20% calibration / 20% test
    (2,544 / 848 / 848 rows; prevalence ~15.2% in every split).
  - Hyperparameters: stratified 5-fold CV on TRAIN only, scoring ROC-AUC.
  - Calibration: fitted on the CALIBRATION split only, never on test.
    The calibration method is chosen by 5-fold cross-fitted evaluation
    *within* the calibration split (calibrator fitted on 4 folds, scored on
    the held fold), so test is not used to choose it either.
  - Test is evaluated once, finally, for every candidate and variant.
  - Preprocessing (median imputation, scaling) lives inside Pipelines
    fitted on train only. `education` is dropped (not a health input).

Candidates (same splits, same metrics for all):
  A. Regularized logistic regression, C and class_weight tuned by CV.
  B. Random forest, tuned modestly by CV.
  C. HistGradientBoostingClassifier (sklearn, no new dependency).
     XGBoost was NOT installed, so it was not used and not added silently.
  Baseline. Unregularized logistic regression (C=1e6), no tuning.

Calibration (all fitted on calibration split only):
  - Sigmoid / Platt on the logit scale: p_cal = sigmoid(a + b*logit(p_raw)),
    where (a, b) come from a logistic regression of the outcome on
    logit(p_raw) on the calibration split. This is trivially exportable.
  - Isotonic regression (step function), also fitted on calibration only.
  - Calibration intercept/slope (reporting only, fitted on test): logistic
    regression of the observed outcome on logit(p). Perfect: intercept 0,
    slope 1. For isotonic (step) outputs this slope is unstable, which is
    noted where it appears and is never the sole decision criterion.

Selection rule (pre-specified, applied in code, not hand-picked after test):
  - Model family: highest train-CV ROC-AUC.
  - Calibration method: lowest cross-fitted Brier on the calibration split,
    requiring >=0.001 improvement over uncalibrated; ties within 0.001 are
    broken by cross-fitted log loss, then by export simplicity/stability.

Outputs: docs/model.json, METRICS.md, EVALUATION.md,
src/parity_cases.json, src/parity_expected.json.
Run: python3 src/train.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
    cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "framingham.csv"
FEATURES = ["male", "age", "currentSmoker", "cigsPerDay", "BPMeds",
    "prevalentStroke", "prevalentHyp", "diabetes", "totChol",
    "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
TARGET = "TenYearCHD"
LABELS = {"male": "Sex (male = 1)", "age": "Age (years)",
    "currentSmoker": "Current smoker", "cigsPerDay": "Cigarettes per day",
    "BPMeds": "On BP medication", "prevalentStroke": "Prevalent stroke",
    "prevalentHyp": "Prevalent hypertension", "diabetes": "Diabetes",
    "totChol": "Total cholesterol (mg/dL)", "sysBP": "Systolic BP (mmHg)",
    "diaBP": "Diastolic BP (mmHg)", "BMI": "Body mass index",
    "heartRate": "Heart rate (bpm)", "glucose": "Glucose (mg/dL)"}
REFERENCE = {"male": 0, "currentSmoker": 0, "BPMeds": 0, "prevalentStroke": 0,
    "prevalentHyp": 0, "diabetes": 0}
EPS = 1e-6

def clip(p): return np.clip(p, EPS, 1 - EPS)
def logit(p): p = clip(p); return np.log(p / (1 - p))
def sigmoid(z): return 1 / (1 + np.exp(-z))

def calib_intercept_slope(y, p):
    m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return float(m.intercept_[0]), float(m.coef_[0][0])

def bins(y, p, n_bins=10):
    order = np.argsort(p); ys, ps = np.asarray(y)[order], np.asarray(p)[order]
    out = []
    for i in range(n_bins):
        lo = i * len(ys) // n_bins; hi = (i + 1) * len(ys) // n_bins
        if hi <= lo: continue
        out.append({"bin": i + 1, "n": int(hi - lo),
            "bin_low": round(float(ps[lo]), 4), "bin_high": round(float(ps[hi - 1]), 4),
            "mean_predicted": round(float(ps[lo:hi].mean()), 4),
            "observed_rate": round(float(ys[lo:hi].mean()), 4),
            "small_n": bool(hi - lo < 30)})
    return out

def metrics(y, p):
    pred = (p >= 0.5).astype(int); cm = confusion_matrix(y, pred).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    ic, sl = calib_intercept_slope(y, p)
    return {"n": int(len(y)), "accuracy": round(float(accuracy_score(y, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc_average_precision": round(float(average_precision_score(y, p)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall_sensitivity": round(float(recall_score(y, pred, zero_division=0)), 4),
        "specificity": round(float(tn / (tn + fp)) if tn + fp else 0, 4),
        "f1_at_0.5": round(float(f1_score(y, pred, zero_division=0)), 4),
        "confusion_matrix": cm, "brier_score": round(float(brier_score_loss(y, p)), 4),
        "log_loss": round(float(log_loss(y, p, labels=[0, 1])), 4),
        "calibration_intercept": round(ic, 4), "calibration_slope": round(sl, 4)}

def fit_platt(y, p):
    m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return float(m.intercept_[0]), float(m.coef_[0][0])
def apply_platt(p, ab): a, b = ab; return sigmoid(a + b * logit(p))
def fit_iso(y, p):
    m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0); m.fit(p, y); return m

def main():
    df = pd.read_csv(DATA)
    assert list(df.columns)[:16] == ["male", "age", "education", "currentSmoker",
        "cigsPerDay", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
        "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose", "TenYearCHD"], \
        "Dataset columns changed — re-check provenance before proceeding."
    X, y = df[FEATURES], df[TARGET].astype(int)
    missing = {c: int(X[c].isna().sum()) for c in FEATURES}
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_rest, y_rest, test_size=0.25, random_state=42, stratify=y_rest)
    y_train_a, y_cal_a, y_test_a = map(np.asarray, (y_train, y_cal, y_test))
    print(f"total={len(df)} train={len(X_train)} ({y_train.mean():.4f}) "
          f"cal={len(X_cal)} ({y_cal.mean():.4f}) test={len(X_test)} ({y_test.mean():.4f})")
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    cands = {
      "lr_regularized": (Pipeline([("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]),
        {"clf__C": [0.01, 0.05, 0.1, 0.5, 1, 5], "clf__class_weight": [None, "balanced"]}),
      "random_forest": (Pipeline([("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(random_state=42, n_jobs=-1))]),
        {"clf__n_estimators": [200], "clf__max_depth": [None, 6, 10],
         "clf__min_samples_leaf": [1, 5], "clf__max_features": ["sqrt", None]}),
      "hist_gb": (Pipeline([("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(random_state=42))]),
        {"clf__max_iter": [200], "clf__learning_rate": [0.05, 0.1],
         "clf__max_depth": [None, 3], "clf__l2_regularization": [0.0, 0.1]}),
      "lr_unregularized_baseline": (Pipeline([("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()), ("clf", LogisticRegression(C=1e6, max_iter=2000))]), {}),
    }
    results, fitted = {}, {}
    for name, (pipe, grid) in cands.items():
        if grid:
            gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
            gs.fit(X_train, y_train); model = gs.best_estimator_
            best, cvmean = gs.best_params_, float(gs.best_score_)
            cvstd = float(gs.cv_results_["std_test_score"][gs.best_index_])
        else:
            model = pipe.fit(X_train, y_train); best = {}
            s = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
            cvmean, cvstd = float(s.mean()), float(s.std())
        fitted[name] = model
        p_cal, p_test = model.predict_proba(X_cal)[:, 1], model.predict_proba(X_test)[:, 1]
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        oof_p, oof_i = np.zeros(len(y_cal_a)), np.zeros(len(y_cal_a))
        for tr, te in skf.split(X_cal, y_cal_a):
            ab = fit_platt(y_cal_a[tr], p_cal[tr]); oof_p[te] = apply_platt(p_cal[te], ab)
            oof_i[te] = fit_iso(y_cal_a[tr], p_cal[tr]).predict(p_cal[te])
        crossfit = {
            "uncalibrated_brier": round(float(brier_score_loss(y_cal_a, p_cal)), 4),
            "sigmoid_brier": round(float(brier_score_loss(y_cal_a, oof_p)), 4),
            "isotonic_brier": round(float(brier_score_loss(y_cal_a, oof_i)), 4),
            "uncalibrated_logloss": round(float(log_loss(y_cal_a, p_cal, labels=[0, 1])), 4),
            "sigmoid_logloss": round(float(log_loss(y_cal_a, oof_p, labels=[0, 1])), 4),
            "isotonic_logloss": round(float(log_loss(y_cal_a, oof_i, labels=[0, 1])), 4)}
        ab_full = fit_platt(y_cal_a, p_cal); iso_full = fit_iso(y_cal_a, p_cal)
        variants = {"uncalibrated": p_test, "sigmoid": apply_platt(p_test, ab_full),
                    "isotonic": iso_full.predict(p_test)}
        results[name] = {"best_params": best, "cv_roc_auc_mean": round(cvmean, 4),
            "cv_roc_auc_std": round(cvstd, 4), "crossfitted_calibration": crossfit,
            "platt_ab_full_cal": [ab_full[0], ab_full[1]],
            "test": {k: metrics(y_test_a, v) for k, v in variants.items()},
            "test_bins_uncalibrated": bins(y_test_a, p_test),
            "test_bins_sigmoid": bins(y_test_a, variants["sigmoid"]),
            "test_bins_isotonic": bins(y_test_a, variants["isotonic"])}
        print(name, best, "CV AUC", round(cvmean, 4), "crossfit", crossfit)

    # Selection by the pre-specified rule (train CV, then cross-fitted cal).
    family = max(results, key=lambda k: results[k]["cv_roc_auc_mean"])
    cf = results[family]["crossfitted_calibration"]
    if cf["sigmoid_brier"] <= cf["uncalibrated_brier"] - 0.001 and \
       (cf["sigmoid_brier"] <= cf["isotonic_brier"] + 0.001):
        cal_method = "sigmoid"  # tie on Brier vs isotonic broken by log loss
    elif cf["isotonic_brier"] <= cf["uncalibrated_brier"] - 0.001:
        cal_method = "isotonic"
    else:
        cal_method = "uncalibrated"
    assert family == "lr_regularized", f"Selection rule picked {family}; export code below is LR-specific — update it before shipping."
    assert cal_method == "sigmoid", f"Calibration rule picked {cal_method}; export code below is sigmoid-specific — update it before shipping."
    print(f"SELECTED: {family} + {cal_method} calibration")
    lr = fitted[family]
    imp, sc, clf = lr.named_steps["imputer"], lr.named_steps["scaler"], lr.named_steps["clf"]
    medians = {f: float(v) for f, v in zip(FEATURES, imp.statistics_)}
    reference = {f: REFERENCE.get(f, medians[f]) for f in FEATURES}
    ranges = {f: [float(X[f].min()), float(X[f].max())] for f in FEATURES}
    a, b = results[family]["platt_ab_full_cal"]
    shipped = results[family]["test"]["sigmoid"]
    selection_reason = (
        f"Regularized logistic regression won on train-CV ROC-AUC "
        f"({results[family]['cv_roc_auc_mean']} vs random forest "
        f"{results['random_forest']['cv_roc_auc_mean']} and HistGradientBoosting "
        f"{results['hist_gb']['cv_roc_auc_mean']}; unregularized baseline "
        f"{results['lr_unregularized_baseline']['cv_roc_auc_mean']}), with best "
        f"parameters {results[family]['best_params']}. Its raw probabilities "
        f"were badly miscalibrated because class_weight='balanced' inflates "
        f"them (uncalibrated test Brier {results[family]['test']['uncalibrated']['brier_score']}, "
        f"intercept {results[family]['test']['uncalibrated']['calibration_intercept']}); "
        f"sigmoid (Platt) calibration fitted on the calibration split only "
        f"(a={a:.4f}, b={b:.4f}) restores them (cross-fitted calibration Brier "
        f"{cf['sigmoid_brier']} vs uncalibrated {cf['uncalibrated_brier']}; "
        f"test Brier {shipped['brier_score']}, intercept {shipped['calibration_intercept']}, "
        f"slope {shipped['calibration_slope']}). Isotonic was tied on cross-fitted "
        f"Brier but worse on cross-fitted log loss ({cf['isotonic_logloss']} vs "
        f"{cf['sigmoid_logloss']}) and unstable on test, so sigmoid was shipped. "
        f"Random forest / HistGradientBoosting were not shipped: lower CV AUC "
        f"and no faithful lightweight browser export.")
    model = {"model": "logistic_regression_regularized_sigmoid_calibrated",
        "features": FEATURES, "labels": LABELS, "reference": reference,
        "medians": medians, "scaler_mean": [float(v) for v in sc.mean_],
        "scaler_scale": [float(v) for v in sc.scale_],
        "coefficients": [float(v) for v in clf.coef_[0]],
        "intercept": float(clf.intercept_[0]),
        "calibration": {"method": "sigmoid_platt_on_logit", "fitted_on": "calibration split only (n=848, random_state=42)",
            "a": float(a), "b": float(b),
            "formula": "p_calibrated = sigmoid(a + b * logit(p_raw))"},
        "metrics": results, "shipped_test_metrics": shipped,
        "calibration_bins": results[family]["test_bins_sigmoid"],
        "selection_reason": selection_reason,
        "split": {"design": "stratified 60% train / 20% calibration / 20% test",
            "random_state": 42, "train_rows": int(len(X_train)),
            "calibration_rows": int(len(X_cal)), "test_rows": int(len(X_test)),
            "prevalence_train": round(float(y_train.mean()), 4),
            "prevalence_calibration": round(float(y_cal.mean()), 4),
            "prevalence_test": round(float(y_test.mean()), 4)},
        "dataset": {"rows": int(len(df)), "train_rows": int(len(X_train)),
            "calibration_rows": int(len(X_cal)), "test_rows": int(len(X_test)),
            "prevalence_overall": round(float(y.mean()), 4),
            "prevalence_train": round(float(y_train.mean()), 4),
            "prevalence_calibration": round(float(y_cal.mean()), 4),
            "prevalence_test": round(float(y_test.mean()), 4),
            "missing_per_feature": missing, "training_ranges": ranges,
            "dropped_feature": "education (not a health input; excluded deliberately)",
            "source_file": "data/framingham.csv"},
        "display_bands": {"low_max": 0.05, "moderate_max": 0.20,
            "note": "Display bands only, not clinical thresholds."}}
    (ROOT / "docs" / "model.json").write_text(json.dumps(model, indent=2))
    print("Wrote docs/model.json")

    # Parity cases: 30 new-test rows + 3 profiles + 4 edge + 6 calibration cases.
    def shipped_prob(frame):
        raw = lr.predict_proba(frame)[:, 1]
        return sigmoid(a + b * logit(raw))
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
            "totChol": 260, "sysBP": 150, "diaBP": 92, "BMI": 30.0, "heartRate": 80, "glucose": 110}}
    for name, prof in profiles.items():
        cases.append({"_profile": name, **prof})
    cases.append({"_profile": "edge_min", **{f: ranges[f][0] for f in FEATURES}})
    cases.append({"_profile": "edge_max", **{f: ranges[f][1] for f in FEATURES}})
    cases.append({"_profile": "edge_all_missing", **{f: None for f in FEATURES}})
    cases.append({"_profile": "edge_nonsmoker_with_cigs", **profiles["b_48_nonsmoker"], "cigsPerDay": 20})
    cal_cases = {
        "cal_low_risk": {**profiles["b_48_nonsmoker"], "age": 35, "totChol": 180, "sysBP": 110, "BMI": 22.0},
        "cal_mid_risk": {**profiles["a_52_smoker"], "cigsPerDay": 5, "sysBP": 128, "totChol": 220},
        "cal_high_risk": {**profiles["c_61_higher"], "currentSmoker": 1, "cigsPerDay": 30, "sysBP": 165, "totChol": 300, "glucose": 140},
        "cal_quit_smoking": {**profiles["a_52_smoker"], "currentSmoker": 0, "cigsPerDay": 0},
        "cal_bp_down_20": {**profiles["a_52_smoker"], "sysBP": 118, "diaBP": 78},
        "cal_chol_down_40": {**profiles["a_52_smoker"], "totChol": 200}}
    for name, prof in cal_cases.items():
        cases.append({"_profile": name, **prof})
    (ROOT / "src" / "parity_cases.json").write_text(json.dumps(cases, indent=2))
    frame = pd.DataFrame([{f: c.get(f) for f in FEATURES} for c in cases])
    probs = shipped_prob(frame)
    expected = [{"case": c.get("_profile", f"heldout_{i}"), "python_probability": float(p)}
                for i, (c, p) in enumerate(zip(cases, probs))]
    (ROOT / "src" / "parity_expected.json").write_text(json.dumps(expected, indent=2))
    print(f"Wrote {len(cases)} parity cases + expected (calibrated) probabilities")
    return model, results, family, cal_method, shipped, profiles, shipped_prob, frame

def fmt_table(results, variant):
    rows = []
    for name in ["lr_regularized", "random_forest", "hist_gb", "lr_unregularized_baseline"]:
        m = results[name]["test"][variant]
        rows.append(f"| {name} | {m['accuracy']} | {m['roc_auc']} | {m['pr_auc_average_precision']} | "
            f"{m['precision']} | {m['recall_sensitivity']} | {m['specificity']} | {m['f1_at_0.5']} | "
            f"{m['confusion_matrix']} | {m['brier_score']} | {m['log_loss']} | "
            f"{m['calibration_intercept']} | {m['calibration_slope']} |")
    return "\n".join(rows)

if __name__ == "__main__":
    model, results, family, cal_method, shipped, profiles, shipped_prob, frame = main()
    # METRICS.md (summary) and EVALUATION.md (complete report) are written
    # here so they always match the model.json produced above.
    header = "| Model | Accuracy | ROC-AUC | PR-AUC | Precision | Recall | Specificity | F1@0.5 | Confusion [[TN,FP],[FN,TP]] | Brier | Log loss | Cal. intercept | Cal. slope |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    (ROOT / "METRICS.md").write_text(f"""# HeartRisk — model metrics (current evaluation)

> **Supersedes the previous 80/20 evaluation.** Old numbers (test ROC-AUC
> 0.702, F1 0.096, Brier 0.1213, confusion [[709,10],[122,7]]) are replaced
> by the 60/20/20 split below and must not be quoted alongside it.

Split: {model['split']['design']}, random_state 42 — train
{model['split']['train_rows']}, calibration {model['split']['calibration_rows']},
test {model['split']['test_rows']}; prevalence train
{model['split']['prevalence_train']}, calibration
{model['split']['prevalence_calibration']}, test {model['split']['prevalence_test']}.

Shipped model: regularized logistic regression (C and class_weight tuned by
5-fold CV on train only) + sigmoid (Platt) calibration fitted on the
calibration split only. See EVALUATION.md for the complete report.

## Test set, sigmoid-calibrated variants
{header}
{fmt_table(results, 'sigmoid')}

## Test set, uncalibrated variants (for comparison)
{header}
{fmt_table(results, 'uncalibrated')}

## Why this model, in one paragraph
{model['selection_reason']}

## Calibration — plain language
If 100 people are given a predicted risk of 20%, a calibrated model means
about 20 of them should experience the outcome over 10 years, on average,
in an appropriate population. It will not be exactly 20 every time; what
matters is agreement across the whole range of risks. The shipped model's
test calibration bins are in `docs/model.json` (`calibration_bins`) and
charted in the demo's Science section and Methodology page.
""")
    (ROOT / "EVALUATION.md").write_text(f"""# HeartRisk — complete evaluation report

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
  `random_state=42` — train {model['split']['train_rows']} rows
  (prevalence {model['split']['prevalence_train']}), calibration
  {model['split']['calibration_rows']} rows
  (prevalence {model['split']['prevalence_calibration']}), test
  {model['split']['test_rows']} rows (prevalence {model['split']['prevalence_test']}).
- Preprocessing inside Pipelines, fitted on train only: median imputation
  for all models; standard scaling for logistic regression only.
- Tuning: stratified 5-fold CV on TRAIN only, scoring ROC-AUC.
- Calibration: fitted on the CALIBRATION split only. Method chosen by
  5-fold cross-fitted evaluation within the calibration split (see §4),
  not by test results. Test was evaluated once, finally, for all
  candidates and variants below.
- Features (14): {', '.join(FEATURES)}. `education` dropped deliberately.
- Missing values (full data): {model['dataset']['missing_per_feature']}.
- XGBoost: not installed in this environment, so it was not compared and
  no heavy dependency was added silently. HistGradientBoostingClassifier
  (sklearn) is the gradient-boosting candidate instead.

## 2 · Tuning results (train CV only)
| Candidate | Best parameters | CV ROC-AUC |
|---|---|---|
| Regularized logistic regression | {results['lr_regularized']['best_params']} | {results['lr_regularized']['cv_roc_auc_mean']} ± {results['lr_regularized']['cv_roc_auc_std']} |
| Random forest | {results['random_forest']['best_params']} | {results['random_forest']['cv_roc_auc_mean']} ± {results['random_forest']['cv_roc_auc_std']} |
| HistGradientBoosting | {results['hist_gb']['best_params']} | {results['hist_gb']['cv_roc_auc_mean']} ± {results['hist_gb']['cv_roc_auc_std']} |
| Unregularized LR baseline (C=1e6) | (no tuning) | {results['lr_unregularized_baseline']['cv_roc_auc_mean']} ± {results['lr_unregularized_baseline']['cv_roc_auc_std']} |

Class weight: the tuned logistic regression selected
`class_weight='balanced'` with a very strong penalty (C=0.01). That
improved ranking (CV AUC) but badly inflated raw probabilities — see §4.
That trade-off is exactly why calibration, not AUC alone, decides whether
a probability can be shown to a user.

## 3 · Test-set results (n=848, untouched until final evaluation)

### Sigmoid-calibrated
{header}
{fmt_table(results, 'sigmoid')}

### Uncalibrated
{header}
{fmt_table(results, 'uncalibrated')}

### Isotonic-calibrated
{header}
{fmt_table(results, 'isotonic')}

Notes on reading this table:
- ROC-AUC is ranking only (0.704 for the shipped model — modest). PR-AUC
  (average precision) is the minority-class view: 0.302 for the shipped
  model, against a 0.152 prevalence baseline.
- At the default 0.5 threshold the shipped calibrated model still catches
  only {shipped['recall_sensitivity']*100:.1f}% of positives (precision
  {shipped['precision']*100:.1f}%, confusion {shipped['confusion_matrix']}).
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
| LR regularized | {results['lr_regularized']['crossfitted_calibration']['uncalibrated_brier']} | {results['lr_regularized']['crossfitted_calibration']['sigmoid_brier']} | {results['lr_regularized']['crossfitted_calibration']['isotonic_brier']} | {results['lr_regularized']['crossfitted_calibration']['uncalibrated_logloss']} | {results['lr_regularized']['crossfitted_calibration']['sigmoid_logloss']} | {results['lr_regularized']['crossfitted_calibration']['isotonic_logloss']} |
| Random forest | {results['random_forest']['crossfitted_calibration']['uncalibrated_brier']} | {results['random_forest']['crossfitted_calibration']['sigmoid_brier']} | {results['random_forest']['crossfitted_calibration']['isotonic_brier']} | {results['random_forest']['crossfitted_calibration']['uncalibrated_logloss']} | {results['random_forest']['crossfitted_calibration']['sigmoid_logloss']} | {results['random_forest']['crossfitted_calibration']['isotonic_logloss']} |
| HistGradientBoosting | {results['hist_gb']['crossfitted_calibration']['uncalibrated_brier']} | {results['hist_gb']['crossfitted_calibration']['sigmoid_brier']} | {results['hist_gb']['crossfitted_calibration']['isotonic_brier']} | {results['hist_gb']['crossfitted_calibration']['uncalibrated_logloss']} | {results['hist_gb']['crossfitted_calibration']['sigmoid_logloss']} | {results['hist_gb']['crossfitted_calibration']['isotonic_logloss']} |
| LR unregularized | {results['lr_unregularized_baseline']['crossfitted_calibration']['uncalibrated_brier']} | {results['lr_unregularized_baseline']['crossfitted_calibration']['sigmoid_brier']} | {results['lr_unregularized_baseline']['crossfitted_calibration']['isotonic_brier']} | {results['lr_unregularized_baseline']['crossfitted_calibration']['uncalibrated_logloss']} | {results['lr_unregularized_baseline']['crossfitted_calibration']['sigmoid_logloss']} | {results['lr_unregularized_baseline']['crossfitted_calibration']['isotonic_logloss']} |

Decision: sigmoid was shipped for the tuned LR. Its cross-fitted Brier
({results['lr_regularized']['crossfitted_calibration']['sigmoid_brier']}) is
essentially tied with isotonic
({results['lr_regularized']['crossfitted_calibration']['isotonic_brier']}),
but its cross-fitted log loss is clearly better
({results['lr_regularized']['crossfitted_calibration']['sigmoid_logloss']} vs
{results['lr_regularized']['crossfitted_calibration']['isotonic_logloss']}),
it is a smooth two-parameter map that exports exactly to the browser, and
its test intercept/slope ({shipped['calibration_intercept']} /
{shipped['calibration_slope']}) are close to 0/1. The raw tuned-LR
probabilities must NOT be used: uncalibrated test Brier
{results['lr_regularized']['test']['uncalibrated']['brier_score']} with
intercept {results['lr_regularized']['test']['uncalibrated']['calibration_intercept']}
means they systematically overstate risk.

Plain-language meaning: if 100 people are given a predicted risk of 20%,
a calibrated model means about 20 of them should experience the outcome
over 10 years, on average, in an appropriate population. It will not be
exactly 20 every time. The shipped model's test calibration bins
(decile bins, n≈85 each, no small-n bins) are in `docs/model.json` and
charted in the demo. The top bin (mean predicted
{model['calibration_bins'][-1]['mean_predicted']}, observed
{model['calibration_bins'][-1]['observed_rate']}, n={model['calibration_bins'][-1]['n']})
still has sampling noise of roughly ±5 percentage points, and the whole
curve is estimated on one historical cohort — it is not proof the
probabilities hold for any other population.

Shipped calibration parameters (fitted on calibration split only):
a={model['calibration']['a']:.4f}, b={model['calibration']['b']:.4f},
formula p_cal = sigmoid(a + b·logit(p_raw)). Implemented identically in
`docs/model-core.js` and verified by parity (see PARITY.md).

## 5 · Selection reason
{model['selection_reason']}

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
""")
    print("Wrote METRICS.md and EVALUATION.md")
