"""
Churn Customer — XGBoost Experiment 2

Improvements over Experiment 1:
  1. Feature engineering — 4 new derived features (same as logistic_experiment_2)
  2. Threshold tuning    — optimise F1 Macro via Precision-Recall curve

scale_pos_weight kept at 4.94 (computed from data).

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
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_curve, classification_report,
)
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation_utils import evaluate_model_cv, print_evaluation


# =========================================================
# 1. Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "dataset" / "E Commerce Dataset.xlsx"

print("=" * 70)
print("Churn Customer — XGBoost Experiment 2 (Feature Engineering + Threshold Tuning)")
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

print(f"\nDataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")


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

print(f"\nLabel distribution: Churn=0 -> {n0:,} ({n0/len(df)*100:.1f}%)  |  Churn=1 -> {n1:,} ({n1/len(df)*100:.1f}%)")
print(f"scale_pos_weight: {scale_pos_weight}")


# =========================================================
# 4. Feature engineering
# =========================================================

print("\nApplying feature engineering...")

def safe_div(a, b, fill=0.0):
    return np.where(b > 0, a / b, fill)

df["AvgCashbackPerOrder"] = safe_div(
    df["CashbackAmount"].fillna(0),
    df["OrderCount"].fillna(0),
)
df["IsHighComplainer"] = (df["Complain"].fillna(0) == 1).astype(int)
df["LowSatisfaction"] = (df["SatisfactionScore"].fillna(3) <= 2).astype(int)

def days_bucket(d):
    if pd.isna(d):
        return "unknown"
    if d <= 7:
        return "recent"
    if d <= 30:
        return "medium"
    return "long"

df["DaysSinceOrderBucket"] = df["DaySinceLastOrder"].apply(days_bucket)

print("  New features: AvgCashbackPerOrder, IsHighComplainer, LowSatisfaction, DaysSinceOrderBucket")


# =========================================================
# 5. Feature selection
# =========================================================

DROP_COLS = [c for c in ["CustomerID", label_col] if c in df.columns]
X = df.drop(columns=DROP_COLS)
y = df[label_col]

numeric_features = [
    "Tenure", "WarehouseToHome", "HourSpendOnApp", "NumberOfDeviceRegistered",
    "SatisfactionScore", "NumberOfAddress", "Complain",
    "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount",
    "DaySinceLastOrder", "CashbackAmount", "CityTier",
    "AvgCashbackPerOrder", "IsHighComplainer", "LowSatisfaction",
]
categorical_features = [
    "PreferredLoginDevice", "PreferredPaymentMode", "Gender",
    "PreferedOrderCat", "MaritalStatus", "DaysSinceOrderBucket",
]

numeric_features = [c for c in numeric_features if c in X.columns]
categorical_features = [c for c in categorical_features if c in X.columns]

print(f"\nNumeric features    ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")


# =========================================================
# 6. Pipeline
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
# 7. Cross-validation — default threshold
# =========================================================

print("\n" + "=" * 70)
print("Step 1: 5-Fold CV — default threshold=0.5...")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = evaluate_model_cv(model, X, y, cv, thresholds=[0.5])
print_evaluation(results, label="XGBoost Experiment 2 — threshold=0.5")

y_proba = results["y_proba"]


# =========================================================
# 8. Threshold tuning — maximise F1 Macro
# =========================================================

print("\n" + "=" * 70)
print("Step 2: Threshold tuning via Precision-Recall curve...")
print("=" * 70)

_, _, thresholds = precision_recall_curve(y, y_proba)

best_threshold = 0.5
best_f1_macro = 0.0

for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    f1_m = f1_score(y, y_pred_t, average="macro", zero_division=0)
    if f1_m > best_f1_macro:
        best_f1_macro = f1_m
        best_threshold = t

print(f"\nOptimal threshold: {best_threshold:.4f}  (F1 Macro = {best_f1_macro:.4f})")

results_tuned = evaluate_model_cv(model, X, y, cv, thresholds=[best_threshold])
print_evaluation(results_tuned, label=f"XGBoost Experiment 2 — threshold={best_threshold:.4f}")

y_pred_tuned = (y_proba >= best_threshold).astype(int)
print("\nClassification Report (optimal threshold):")
print(classification_report(y, y_pred_tuned, zero_division=0))


# =========================================================
# 9. Threshold sweep
# =========================================================

print("\nThreshold sweep (Recall vs Precision tradeoff):")
print(f"{'Threshold':>10} {'Precision(1)':>13} {'Recall(1)':>10} {'F1(1)':>8} {'F1 Macro':>10}")

for t in np.arange(0.1, 0.91, 0.05):
    y_t = (y_proba >= t).astype(int)
    p1 = precision_score(y, y_t, pos_label=1, zero_division=0)
    r1 = recall_score(y, y_t, pos_label=1, zero_division=0)
    f1 = f1_score(y, y_t, pos_label=1, zero_division=0)
    fm = f1_score(y, y_t, average="macro", zero_division=0)
    marker = " << optimal" if abs(t - best_threshold) < 0.025 else ""
    print(f"{t:>10.2f} {p1:>13.4f} {r1:>10.4f} {f1:>8.4f} {fm:>10.4f}{marker}")


# =========================================================
# 10. Save summary
# =========================================================

m_default = results["thresholds"][0.5]
m_tuned = results_tuned["thresholds"][best_threshold]

summary = {
    "scale_pos_weight": scale_pos_weight,
    "optimal_threshold": round(float(best_threshold), 4),
    # default threshold
    "accuracy_default": round(m_default["accuracy"], 6),
    "balanced_accuracy_default": round(m_default["balanced_accuracy"], 6),
    "precision_1_default": round(m_default["precision_1"], 6),
    "recall_1_default": round(m_default["recall_1"], 6),
    "f1_1_default": round(m_default["f1_1"], 6),
    "f1_macro_default": round(m_default["f1_macro"], 6),
    "roc_auc": round(m_default["roc_auc"], 6),
    "pr_auc": round(m_default["pr_auc"], 6),
    # optimal threshold
    "accuracy_tuned": round(m_tuned["accuracy"], 6),
    "balanced_accuracy_tuned": round(m_tuned["balanced_accuracy"], 6),
    "precision_1_tuned": round(m_tuned["precision_1"], 6),
    "recall_1_tuned": round(m_tuned["recall_1"], 6),
    "f1_1_tuned": round(m_tuned["f1_1"], 6),
    "f1_macro_tuned": round(m_tuned["f1_macro"], 6),
}

SUMMARY_FILE = BASE_DIR / "xgboost_experiment_2_summary.csv"
pd.DataFrame([summary], index=["xgboost_experiment_2"]).to_csv(SUMMARY_FILE, index=True)
print(f"\nSummary saved: {SUMMARY_FILE}")

print("\n" + "=" * 70)
print("Experiment 2 complete.")
print("=" * 70)
