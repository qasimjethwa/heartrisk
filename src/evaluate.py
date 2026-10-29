"""Exploratory evaluation (no files overwritten except /tmp output).

Split design (supersedes old 80/20): stratified 60% train / 20% calibration /
20% test, random_state=42. Preprocessing inside Pipelines fitted on train
only. Tuning by stratified 5-fold CV on train only. Calibration fitted on
the calibration split only; calibration-method choice by 5-fold cross-fitted
evaluation *within* the calibration split (so test is not used to choose it).
Test is evaluated once, finally, for every candidate/variant.
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["male","age","currentSmoker","cigsPerDay","BPMeds","prevalentStroke",
    "prevalentHyp","diabetes","totChol","sysBP","diaBP","BMI","heartRate","glucose"]
TARGET = "TenYearCHD"
EPS = 1e-6

def clip(p): return np.clip(p, EPS, 1-EPS)
def logit(p): p=clip(p); return np.log(p/(1-p))
def sigmoid(z): return 1/(1+np.exp(-z))

def calib_intercept_slope(y, p):
    x = logit(p).reshape(-1,1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(x, y)
    return float(m.intercept_[0]), float(m.coef_[0][0])

def bins(y, p, n_bins=10):
    order = np.argsort(p); ys, ps = np.asarray(y)[order], np.asarray(p)[order]
    out=[]
    for i in range(n_bins):
        lo = i*len(ys)//n_bins; hi = (i+1)*len(ys)//n_bins
        if hi<=lo: continue
        out.append({"bin": i+1, "n": int(hi-lo),
            "bin_low": round(float(ps[lo]),4), "bin_high": round(float(ps[hi-1]),4),
            "mean_predicted": round(float(ps[lo:hi].mean()),4),
            "observed_rate": round(float(ys[lo:hi].mean()),4)})
    return out

def metrics(y, p):
    pred=(p>=0.5).astype(int); cm=confusion_matrix(y,pred).tolist()
    tn,fp,fn,tp=cm[0][0],cm[0][1],cm[1][0],cm[1][1]
    return {"n":int(len(y)), "accuracy":round(float(accuracy_score(y,pred)),4),
        "roc_auc":round(float(roc_auc_score(y,p)),4),
        "pr_auc_average_precision":round(float(average_precision_score(y,p)),4),
        "precision":round(float(precision_score(y,pred,zero_division=0)),4),
        "recall_sensitivity":round(float(recall_score(y,pred,zero_division=0)),4),
        "specificity":round(float(tn/(tn+fp)) if tn+fp else 0,4),
        "f1_at_0.5":round(float(f1_score(y,pred,zero_division=0)),4),
        "confusion_matrix":cm, "brier_score":round(float(brier_score_loss(y,p)),4),
        "log_loss":round(float(log_loss(y,p,labels=[0,1])),4),
        "calibration_intercept":round(float(calib_intercept_slope(y,p)[0]),4),
        "calibration_slope":round(float(calib_intercept_slope(y,p)[1]),4)}

def fit_platt(y, p):
    m=LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1,1), y)
    return float(m.intercept_[0]), float(m.coef_[0][0])
def apply_platt(p, ab):
    a,b=ab; return sigmoid(a+b*logit(p))
def fit_iso(y, p):
    m=IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0); m.fit(p,y); return m

def main():
    df=pd.read_csv(ROOT/"data"/"framingham.csv")
    X,y=df[FEATURES],df[TARGET].astype(int)
    X_rest,X_test,y_rest,y_test=train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
    X_train,X_cal,y_train,y_cal=train_test_split(X_rest,y_rest,test_size=0.25,random_state=42,stratify=y_rest)
    print(f"total={len(df)} train={len(X_train)} ({y_train.mean():.4f}) cal={len(X_cal)} ({y_cal.mean():.4f}) test={len(X_test)} ({y_test.mean():.4f})")
    cv=StratifiedKFold(5,shuffle=True,random_state=42)
    cands={
      "lr_regularized": (Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("clf",LogisticRegression(max_iter=2000))]),
        {"clf__C":[0.01,0.05,0.1,0.5,1,5],"clf__class_weight":[None,"balanced"]}),
      "random_forest": (Pipeline([("imputer",SimpleImputer(strategy="median")),("clf",RandomForestClassifier(random_state=42,n_jobs=-1))]),
        {"clf__n_estimators":[200],"clf__max_depth":[None,6,10],"clf__min_samples_leaf":[1,5],"clf__max_features":["sqrt",None]}),
      "hist_gb": (Pipeline([("imputer",SimpleImputer(strategy="median")),("clf",HistGradientBoostingClassifier(random_state=42))]),
        {"clf__max_iter":[200],"clf__learning_rate":[0.05,0.1],"clf__max_depth":[None,3],"clf__l2_regularization":[0.0,0.1]}),
      "lr_unregularized_baseline": (Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("clf",LogisticRegression(C=1e6,max_iter=2000))]), {}),
    }
    results={}
    for name,(pipe,grid) in cands.items():
        if grid:
            gs=GridSearchCV(pipe,grid,cv=cv,scoring="roc_auc",n_jobs=-1); gs.fit(X_train,y_train)
            model=gs.best_estimator_; best=gs.best_params_; cvmean=float(gs.best_score_)
            # CV std across folds for best params: refit score via cv_results best index std
            cvstd=float(gs.cv_results_["std_test_score"][gs.best_index_])
        else:
            model=pipe.fit(X_train,y_train); best={}; 
            from sklearn.model_selection import cross_val_score
            s=cross_val_score(model,X_train,y_train,cv=cv,scoring="roc_auc"); cvmean=float(s.mean()); cvstd=float(s.std())
        p_cal=model.predict_proba(X_cal)[:,1]; p_test=model.predict_proba(X_test)[:,1]
        # cross-fitted calibration within cal split
        skf=StratifiedKFold(5,shuffle=True,random_state=42)
        oof_platt=np.zeros(len(y_cal)); oof_iso=np.zeros(len(y_cal))
        for tr,te in skf.split(X_cal,y_cal):
            ab=fit_platt(np.asarray(y_cal)[tr], p_cal[tr]); oof_platt[te]=apply_platt(p_cal[te],ab)
            iso=fit_iso(np.asarray(y_cal)[tr], p_cal[tr]); oof_iso[te]=iso.predict(p_cal[te])
        crossfit={"uncalibrated_brier":round(float(brier_score_loss(y_cal,p_cal)),4),
            "sigmoid_brier":round(float(brier_score_loss(y_cal,oof_platt)),4),
            "isotonic_brier":round(float(brier_score_loss(y_cal,oof_iso)),4),
            "uncalibrated_logloss":round(float(log_loss(y_cal,p_cal,labels=[0,1])),4),
            "sigmoid_logloss":round(float(log_loss(y_cal,oof_platt,labels=[0,1])),4),
            "isotonic_logloss":round(float(log_loss(y_cal,oof_iso,labels=[0,1])),4)}
        # full-cal calibrators, applied to test
        ab_full=fit_platt(np.asarray(y_cal),p_cal); iso_full=fit_iso(np.asarray(y_cal),p_cal)
        variants={"uncalibrated":p_test, "sigmoid":apply_platt(p_test,ab_full), "isotonic":iso_full.predict(p_test)}
        testm={k:metrics(np.asarray(y_test),v) for k,v in variants.items()}
        results[name]={"best_params":best,"cv_roc_auc_mean":round(cvmean,4),"cv_roc_auc_std":round(cvstd,4),
            "crossfitted_calibration":crossfit,"platt_ab_full_cal":ab_full,
            "test":testm, "test_bins_uncalibrated":bins(np.asarray(y_test),p_test),
            "test_bins_sigmoid":bins(np.asarray(y_test),variants["sigmoid"]),
            "test_bins_isotonic":bins(np.asarray(y_test),variants["isotonic"])}
        print(name, "best", best, "cv", round(cvmean,4), "crossfit", crossfit)
        for k,v in testm.items(): print("  test",k,v)
    Path("/tmp/eval_results.json").write_text(json.dumps(results,indent=2,default=str))
    print("Wrote /tmp/eval_results.json")

if __name__=="__main__": main()
