"""
Churn Customer — CatBoost Experiment 1

Baseline CatBoost model:
  - Native categorical handling (no OHE needed)
  - Median imputation for numerics, mode imputation for categoricals
  - No threshold tuning, no class weighting

Split     : Hold-out 70% train / 30% test, shuffle=False
            (last 30% of rows become test — original order preserved)
Exposure  : Test set seen exactly once, at final evaluation only.

Data source  : E Commerce Dataset.xlsx  (Sheet: E Comm)
Target       : Churn  (1 = churned, 0 = retained)
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from catboost import CatBoostClassifier


# ---------------------------------------------------------------------------
# evaluation_utils  (inline)
# ---------------------------------------------------------------------------

def evaluate_holdout(model, X_test, y_test, thresholds=None, pos_label=1):
    if thresholds is None:
        thresholds = [0.5]
    y_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc  = average_precision_score(y_test, y_proba, pos_label=pos_label)
    threshold_results = {}
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        threshold_results[t] = {
            "accuracy":          accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "precision_1":       precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0),
            "recall_1":          recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0),
            "f1_1":              f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0),
            "f1_macro":          f1_score(y_test, y_pred, average="macro", zero_division=0),
            "roc_auc":           roc_auc,
            "pr_auc":            pr_auc,
            "confusion_matrix":  confusion_matrix(y_test, y_pred),
        }
    return {"thresholds": threshold_results, "y_proba": y_proba}


def print_evaluation(results, label=""):
    header = "Evaluation Results" + (f" — {label}" if label else "")
    print("\n" + "=" * 70)
    print(header)
    print("=" * 70)
    for threshold, metrics in results["thresholds"].items():
        print(f"\n  Threshold = {threshold:.2f}")
        print(f"  {'-' * 40}")
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  Precision (Ch=1):  {metrics['precision_1']:.4f}")
        print(f"  Recall    (Ch=1):  {metrics['recall_1']:.4f}")
        print(f"  F1        (Ch=1):  {metrics['f1_1']:.4f}")
        print(f"  F1 Macro:          {metrics['f1_macro']:.4f}")
        print(f"  ROC AUC:           {metrics['roc_auc']:.4f}")
        print(f"  PR AUC:            {metrics['pr_auc']:.4f}")
        print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
        print(f"               Pred 0   Pred 1")
        cm = metrics["confusion_matrix"]
        print(f"  Actual 0  :  {cm[0][0]:>6}   {cm[0][1]:>6}")
        print(f"  Actual 1  :  {cm[1][0]:>6}   {cm[1][1]:>6}")


# ---------------------------------------------------------------------------
# sklearn-cloneable CatBoost wrapper
# ---------------------------------------------------------------------------

class CloneableCatBoost(CatBoostClassifier):
    def __init__(self, cat_features_list=None, **kwargs):
        self.cat_features_list = cat_features_list
        super().__init__(cat_features=cat_features_list, **kwargs)

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params["cat_features_list"] = self.cat_features_list
        params.pop("cat_features", None)
        return params


# ===========================================================================
# 1. Paths
# ===========================================================================

BASE_DIR  = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "dataset" / "E Commerce Dataset.xlsx"

_UPLOAD_PATH = Path("/mnt/user-data/uploads/E_Commerce_Dataset.xlsx")
if not DATA_FILE.exists() and _UPLOAD_PATH.exists():
    DATA_FILE = _UPLOAD_PATH

print("=" * 70)
print("Churn Customer — CatBoost Experiment 1 (Baseline)")
print("Split: Hold-out 70/30, shuffle=False")
print("=" * 70)
print(f"\nDataset: {DATA_FILE}")

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")


# ===========================================================================
# 2. Load & validate
# ===========================================================================

df = pd.read_excel(DATA_FILE, sheet_name="E Comm")
df.columns = df.columns.str.strip()

print(f"\nDataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

LABEL = "Churn"
df[LABEL] = pd.to_numeric(df[LABEL], errors="coerce")
missing_labels = df[LABEL].isna().sum()
if missing_labels > 0:
    print(f"\nWarning: {missing_labels} rows with missing label — dropping them.")
    df = df.dropna(subset=[LABEL])
df[LABEL] = df[LABEL].astype(int)

n0 = (df[LABEL] == 0).sum()
n1 = (df[LABEL] == 1).sum()
print(f"\nLabel distribution (full dataset):")
print(f"  Churn = 0 (retained): {n0:,} ({n0/len(df)*100:.1f}%)")
print(f"  Churn = 1 (churned):  {n1:,} ({n1/len(df)*100:.1f}%)")
print(f"  Imbalance ratio: {n0/n1:.2f}:1")


# ===========================================================================
# 3. Split — 70/30, shuffle=False
# ===========================================================================

NUMERIC_FEATURES = [
    "Tenure", "WarehouseToHome", "HourSpendOnApp", "NumberOfDeviceRegistered",
    "SatisfactionScore", "NumberOfAddress", "Complain",
    "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount",
    "DaySinceLastOrder", "CashbackAmount", "CityTier",
]
CATEGORICAL_FEATURES = [
    "PreferredLoginDevice", "PreferredPaymentMode", "Gender",
    "PreferedOrderCat", "MaritalStatus",
]
NUMERIC_FEATURES     = [c for c in NUMERIC_FEATURES     if c in df.columns]
CATEGORICAL_FEATURES = [c for c in CATEGORICAL_FEATURES if c in df.columns]

X = df.drop(columns=["CustomerID", LABEL])
y = df[LABEL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42,
)

print(f"\nSplit (stratify=y, random_state=42):")
print(f"  Train: {len(X_train):,} rows ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Test : {len(X_test):,}  rows ({len(X_test)/len(X)*100:.1f}%)")
print(f"\n  Train — Churn=1: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
print(f"  Test  — Churn=1: {y_test.sum():,}  ({y_test.mean()*100:.1f}%)")
print(f"\nNumeric features     ({len(NUMERIC_FEATURES)}): {NUMERIC_FEATURES}")
print(f"Categorical features ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")


# ===========================================================================
# 4. Pipeline
# ===========================================================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("to_str",  FunctionTransformer(
        lambda arr: arr.astype(str), feature_names_out="one-to-one"
    )),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ],
    remainder="drop",
)

cat_indices = list(range(len(NUMERIC_FEATURES),
                         len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)))

catboost_clf = CloneableCatBoost(
    cat_features_list=cat_indices,
    iterations=500,
    learning_rate=0.05,
    depth=6,
    eval_metric="AUC",
    random_seed=42,
    verbose=0,
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier",   catboost_clf),
])

print(f"\nCatBoost config:")
print(f"  iterations    = 500")
print(f"  learning_rate = 0.05")
print(f"  depth         = 6")


# ===========================================================================
# 5. Train
# ===========================================================================

print("\n" + "=" * 70)
print("Training on 70% train set...")
print("=" * 70)

model.fit(X_train, y_train)

print("Training complete.")


# ===========================================================================
# 6. Evaluate — test set (חשיפה יחידה)
# ===========================================================================

print("\n" + "=" * 70)
print("Evaluating on 30% test set (single exposure)...")
print("=" * 70)

results = evaluate_holdout(model, X_test, y_test, thresholds=[0.5])
print_evaluation(results, label="CatBoost Experiment 1 (Baseline) — Test Set")


# ===========================================================================
# 7. Save summary
# ===========================================================================

metrics_05 = results["thresholds"][0.5]

summary = {
    "oof_accuracy":          round(metrics_05["accuracy"],          6),
    "oof_balanced_accuracy": round(metrics_05["balanced_accuracy"], 6),
    "oof_precision_1":       round(metrics_05["precision_1"],       6),
    "oof_recall_1":          round(metrics_05["recall_1"],          6),
    "oof_f1_1":              round(metrics_05["f1_1"],              6),
    "oof_f1_macro":          round(metrics_05["f1_macro"],          6),
    "oof_roc_auc":           round(metrics_05["roc_auc"],           6),
    "oof_pr_auc":            round(metrics_05["pr_auc"],            6),
}

SUMMARY_FILE = BASE_DIR / "catboost_experiment_1_summary.csv"
pd.DataFrame([summary], index=["catboost_experiment_1"]).to_csv(SUMMARY_FILE)
print(f"\nSummary saved → {SUMMARY_FILE}")

print("\n" + "=" * 70)
print("Experiment 1 complete.")
print("=" * 70)