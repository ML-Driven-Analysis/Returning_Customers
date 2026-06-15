"""
Churn Customer — CatBoost Experiment 2

Changes vs Experiment 1:
  - auto_class_weights="Balanced"  → penalises churn=1 misses more
  - Threshold tuning: evaluate at [0.3, 0.4, 0.5] to boost Recall (Ch=1)

Data source  : E Commerce Dataset.xlsx  (Sheet: E Comm)
Target       : Churn  (1 = churned, 0 = retained)
Validation   : StratifiedKFold (5 splits, shuffle=True, random_state=42)
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from catboost import CatBoostClassifier

from sklearn.model_selection import cross_val_predict as _cvp
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)


# ---------------------------------------------------------------------------
# evaluation_utils  (inline)
# ---------------------------------------------------------------------------

def evaluate_model_cv(model, X, y, cv, thresholds=None, pos_label=1):
    if thresholds is None:
        thresholds = [0.5]
    y_proba = _cvp(model, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    y_oof   = _cvp(model, X, y, cv=cv, method="predict",       n_jobs=1)
    roc_auc = roc_auc_score(y, y_proba)
    pr_auc  = average_precision_score(y, y_proba, pos_label=pos_label)
    threshold_results = {}
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        threshold_results[t] = {
            "accuracy":          accuracy_score(y, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y, y_pred),
            "precision_1":       precision_score(y, y_pred, pos_label=pos_label, zero_division=0),
            "recall_1":          recall_score(y, y_pred, pos_label=pos_label, zero_division=0),
            "f1_1":              f1_score(y, y_pred, pos_label=pos_label, zero_division=0),
            "f1_macro":          f1_score(y, y_pred, average="macro", zero_division=0),
            "roc_auc":           roc_auc,
            "pr_auc":            pr_auc,
            "confusion_matrix":  confusion_matrix(y, y_pred),
        }
    return {"thresholds": threshold_results, "y_proba": y_proba, "y_oof": y_oof}


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
# sklearn-cloneable CatBoost wrapper  (same as Experiment 1)
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
print("Churn Customer — CatBoost Experiment 2")
print("Changes: auto_class_weights='Balanced' + threshold tuning [0.3, 0.4, 0.5]")
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
df = df.dropna(subset=[LABEL])
df[LABEL] = df[LABEL].astype(int)

n0 = (df[LABEL] == 0).sum()
n1 = (df[LABEL] == 1).sum()
print(f"\nLabel distribution:")
print(f"  Churn = 0 (retained): {n0:,} ({n0/len(df)*100:.1f}%)")
print(f"  Churn = 1 (churned):  {n1:,} ({n1/len(df)*100:.1f}%)")
print(f"  Imbalance ratio: {n0/n1:.2f}:1")


# ===========================================================================
# 3. Features
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

# ── KEY CHANGE: auto_class_weights="Balanced" ──────────────────────────────
catboost_clf = CloneableCatBoost(
    cat_features_list=cat_indices,
    iterations=500,
    learning_rate=0.05,
    depth=6,
    eval_metric="AUC",
    auto_class_weights="Balanced",   # <-- NEW
    random_seed=42,
    verbose=0,
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier",   catboost_clf),
])

print(f"\nCatBoost config:")
print(f"  iterations        = 500")
print(f"  learning_rate     = 0.05")
print(f"  depth             = 6")
print(f"  auto_class_weights= Balanced   ← NEW")
print(f"  cat_features      = {CATEGORICAL_FEATURES}")
print(f"\nThreshold sweep    = [0.30, 0.40, 0.50]   ← NEW")


# ===========================================================================
# 5. Cross-validation
# ===========================================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "=" * 70)
print("Running 5-Fold Stratified Cross Validation...")
print("=" * 70)

THRESHOLDS = [0.30, 0.40, 0.50]

results = evaluate_model_cv(
    model, X, y, cv=cv,
    thresholds=THRESHOLDS,
    pos_label=1,
)

print_evaluation(results, label="CatBoost Experiment 2 (class_weight + threshold tuning)")


# ===========================================================================
# 6. Comparison vs Experiment 1  (threshold=0.5 baseline)
# ===========================================================================

EXP1 = {
    "recall_1":   0.7753,
    "precision_1": 0.9269,
    "f1_1":       0.8443,
    "f1_macro":   0.9079,
    "roc_auc":    0.9835,
}

print("\n" + "=" * 70)
print("Comparison vs Experiment 1  (threshold=0.50)")
print("=" * 70)
print(f"  {'Metric':<22} {'Exp 1':>8} {'Exp 2 (t=0.50)':>15} {'Exp 2 (t=0.40)':>15} {'Exp 2 (t=0.30)':>15}")
print(f"  {'-'*75}")

metrics_to_compare = ["recall_1", "precision_1", "f1_1", "f1_macro", "roc_auc"]
for m in metrics_to_compare:
    exp1_val = EXP1[m]
    row = f"  {m:<22} {exp1_val:>8.4f}"
    for t in [0.50, 0.40, 0.30]:
        val = results["thresholds"][t][m]
        delta = val - exp1_val
        sign  = "+" if delta >= 0 else ""
        row += f"   {val:.4f}({sign}{delta:.4f})"
    print(row)


# ===========================================================================
# 7. Save summary
# ===========================================================================

rows = []
for t in THRESHOLDS:
    m = results["thresholds"][t]
    rows.append({
        "experiment":        f"catboost_experiment_2_t{int(t*100):02d}",
        "threshold":         t,
        "oof_accuracy":      round(m["accuracy"],          6),
        "oof_bal_accuracy":  round(m["balanced_accuracy"], 6),
        "oof_precision_1":   round(m["precision_1"],       6),
        "oof_recall_1":      round(m["recall_1"],          6),
        "oof_f1_1":          round(m["f1_1"],              6),
        "oof_f1_macro":      round(m["f1_macro"],          6),
        "oof_roc_auc":       round(m["roc_auc"],           6),
        "oof_pr_auc":        round(m["pr_auc"],            6),
    })

SUMMARY_FILE = BASE_DIR / "catboost_experiment_2_summary.csv"
pd.DataFrame(rows).set_index("experiment").to_csv(SUMMARY_FILE)
print(f"\nSummary saved → {SUMMARY_FILE}")

print("\n" + "=" * 70)
print("Experiment 2 complete.")
print("=" * 70)