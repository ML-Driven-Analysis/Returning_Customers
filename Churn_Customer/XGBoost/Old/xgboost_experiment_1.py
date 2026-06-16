"""
Churn Customer — XGBoost Experiment 1

Baseline XGBoost with imbalance handling via scale_pos_weight.
No feature engineering, no threshold tuning — clean comparison point
against logistic_experiment_1.

Imbalance ratio: 4682 / 948 ≈ 4.94  →  scale_pos_weight=4.94

Data source: Churn_Customer/dataset/E Commerce Dataset.xlsx (Sheet: E Comm)
Target: Churn (1 = churned, 0 = retained)

Validation: StratifiedKFold (5 splits, shuffle=True, random_state=42)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation_utils import evaluate_model_cv, print_evaluation


# =========================================================
# 1. Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "dataset" / "E Commerce Dataset.xlsx"

print("=" * 70)
print("Churn Customer — XGBoost Experiment 1 (Baseline + scale_pos_weight)")
print("=" * 70)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "See README_data_extraction.md for download instructions."
    )


# =========================================================
# 2. Load data
# =========================================================

df = pd.read_excel(DATA_FILE, sheet_name="E Comm")
df.columns = df.columns.str.strip()

print(f"\nDataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")


# =========================================================
# 3. Label
# =========================================================

label_col = "Churn"
df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
df = df.dropna(subset=[label_col])
df[label_col] = df[label_col].astype(int)

n0 = (df[label_col] == 0).sum()
n1 = (df[label_col] == 1).sum()
scale_pos_weight = round(n0 / n1, 2)

print(f"\nLabel distribution:")
print(f"  Churn=0 (retained): {n0:,} ({n0/len(df)*100:.1f}%)")
print(f"  Churn=1 (churned):  {n1:,} ({n1/len(df)*100:.1f}%)")
print(f"  scale_pos_weight:   {scale_pos_weight}")


# =========================================================
# 4. Feature selection (same as logistic_experiment_1)
# =========================================================

DROP_COLS = [c for c in ["CustomerID", label_col] if c in df.columns]
X = df.drop(columns=DROP_COLS)
y = df[label_col]

numeric_features = [
    "Tenure", "WarehouseToHome", "HourSpendOnApp", "NumberOfDeviceRegistered",
    "SatisfactionScore", "NumberOfAddress", "Complain",
    "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount",
    "DaySinceLastOrder", "CashbackAmount", "CityTier",
]
categorical_features = [
    "PreferredLoginDevice", "PreferredPaymentMode", "Gender",
    "PreferedOrderCat", "MaritalStatus",
]

numeric_features = [c for c in numeric_features if c in X.columns]
categorical_features = [c for c in categorical_features if c in X.columns]

print(f"\nNumeric features    ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")


# =========================================================
# 5. Pipeline
# =========================================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )),
])


# =========================================================
# 6. Cross-validation evaluation
# =========================================================

print("\n" + "=" * 70)
print("Running 5-Fold Stratified Cross-Validation...")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = evaluate_model_cv(model, X, y, cv, thresholds=[0.5])
print_evaluation(results, label="XGBoost Experiment 1 — threshold=0.5")


# =========================================================
# 7. Threshold sweep
# =========================================================

print("\n" + "=" * 70)
print("Threshold sweep (Recall vs Precision tradeoff):")
print("=" * 70)
print(f"{'Threshold':>10} {'Precision(1)':>13} {'Recall(1)':>10} {'F1(1)':>8} {'F1 Macro':>10}")

y_proba = results["y_proba"]
for t in np.arange(0.1, 0.91, 0.05):
    from sklearn.metrics import precision_score, recall_score, f1_score
    y_t = (y_proba >= t).astype(int)
    p1 = precision_score(y, y_t, pos_label=1, zero_division=0)
    r1 = recall_score(y, y_t, pos_label=1, zero_division=0)
    f1 = f1_score(y, y_t, pos_label=1, zero_division=0)
    fm = f1_score(y, y_t, average="macro", zero_division=0)
    print(f"{t:>10.2f} {p1:>13.4f} {r1:>10.4f} {f1:>8.4f} {fm:>10.4f}")


# =========================================================
# 8. Save summary
# =========================================================

m = results["thresholds"][0.5]
summary = {
    "scale_pos_weight": scale_pos_weight,
    "accuracy": round(m["accuracy"], 6),
    "balanced_accuracy": round(m["balanced_accuracy"], 6),
    "precision_1": round(m["precision_1"], 6),
    "recall_1": round(m["recall_1"], 6),
    "f1_1": round(m["f1_1"], 6),
    "f1_macro": round(m["f1_macro"], 6),
    "roc_auc": round(m["roc_auc"], 6),
    "pr_auc": round(m["pr_auc"], 6),
}

SUMMARY_FILE = BASE_DIR / "xgboost_experiment_1_summary.csv"
pd.DataFrame([summary], index=["xgboost_experiment_1"]).to_csv(SUMMARY_FILE, index=True)
print(f"\nSummary saved: {SUMMARY_FILE}")

print("\n" + "=" * 70)
print("Experiment 1 complete.")
print("=" * 70)
