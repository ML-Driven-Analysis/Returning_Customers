"""
Churn Customer — CatBoost Experiment 1

Baseline CatBoost model:
  - Native categorical handling via CatBoost (no OHE needed)
  - Median imputation for numerics, mode imputation for categoricals
  - No threshold tuning, no class weighting
  - Uses shared evaluation_utils framework

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

# ---------------------------------------------------------------------------
# evaluation_utils  (inline — same interface as the shared module)
# ---------------------------------------------------------------------------
from sklearn.model_selection import cross_val_predict as _cvp
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)


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
        print(f"\n  Threshold = {threshold:.4f}")
        print(f"  {'-' * 40}")
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  Precision (Ch=1):  {metrics['precision_1']:.4f}")
        print(f"  Recall    (Ch=1):  {metrics['recall_1']:.4f}")
        print(f"  F1        (Ch=1):  {metrics['f1_1']:.4f}")
        print(f"  F1 Macro:          {metrics['f1_macro']:.4f}")
        print(f"  ROC AUC:           {metrics['roc_auc']:.4f}")
        print(f"  PR AUC:            {metrics['pr_auc']:.4f}")
        print(f"\n  Confusion Matrix:")
        for row in metrics["confusion_matrix"]:
            print(f"    {row}")


# ---------------------------------------------------------------------------
# sklearn-cloneable CatBoost wrapper
# ---------------------------------------------------------------------------
# sklearn's cross_val_predict calls clone() on the estimator each fold.
# CatBoostClassifier modifies cat_features inside __init__, which breaks
# clone(). This thin wrapper stores cat_features_list as a plain constructor
# argument so clone() can reconstruct it correctly.
# ---------------------------------------------------------------------------

class CloneableCatBoost(CatBoostClassifier):
    """CatBoostClassifier that survives sklearn.base.clone()."""

    def __init__(self, cat_features_list=None, **kwargs):
        self.cat_features_list = cat_features_list
        super().__init__(cat_features=cat_features_list, **kwargs)

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params["cat_features_list"] = self.cat_features_list
        params.pop("cat_features", None)   # avoid duplicate kwarg on re-init
        return params



# Paths


BASE_DIR  = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR.parent / "dataset" / "E Commerce Dataset.xlsx"

_UPLOAD_PATH = Path("/mnt/user-data/uploads/E_Commerce_Dataset.xlsx")
if not DATA_FILE.exists() and _UPLOAD_PATH.exists():
    DATA_FILE = _UPLOAD_PATH

print("=" * 70)
print("Churn Customer — CatBoost Experiment 1 (Baseline)")
print("=" * 70)
print(f"\nDataset: {DATA_FILE}")

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")


# ===========================================================================
# 2. Load & validate data   (identical to Experiment 1)
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
print(f"\nLabel distribution:")
print(f"  Churn = 0 (retained): {n0:,} ({n0/len(df)*100:.1f}%)")
print(f"  Churn = 1 (churned):  {n1:,} ({n1/len(df)*100:.1f}%)")
print(f"  Imbalance ratio: {n0/n1:.2f}:1")


# ===========================================================================
# 3. Feature definitions   (same split as Experiment 1)
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

print(f"\nNumeric features     ({len(NUMERIC_FEATURES)}): {NUMERIC_FEATURES}")
print(f"Categorical features ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")

X = df.drop(columns=["CustomerID", LABEL])
y = df[LABEL]


# ===========================================================================
# 4. Preprocessing pipeline
# ===========================================================================
# CatBoost handles categoricals natively — no StandardScaler or OHE needed.
# We only impute and cast cats to str so CatBoost never receives a float NaN
# in a categorical column.
#
# ColumnTransformer output layout:
#   [ numeric_cols (13) | categorical_cols (5) ]
# Cat feature indices in the transformed matrix = [13, 14, 15, 16, 17]
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

# Categorical feature indices in the post-transform matrix
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
print(f"  cat_features  = indices {cat_indices}  → {CATEGORICAL_FEATURES}")


# ===========================================================================
# 5. Cross-validation via evaluate_model_cv
# ===========================================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "=" * 70)
print("Running 5-Fold Stratified Cross Validation...")
print("=" * 70)

results = evaluate_model_cv(
    model, X, y, cv=cv,
    thresholds=[0.5],
    pos_label=1,
)

print_evaluation(results, label="CatBoost Experiment 1 (Baseline)")


# ===========================================================================
# 6. Save summary CSV   (same convention as Experiment 1)
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