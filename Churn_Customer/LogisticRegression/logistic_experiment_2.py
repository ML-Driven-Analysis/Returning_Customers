"""
Churn Customer — Logistic Regression Experiment 2

Improvements over Experiment 1:
  1. class_weight='balanced'  — addresses 83/17 class imbalance
  2. Feature engineering      — 4 new derived features
  3. Threshold tuning         — Precision-Recall curve, optimise F1 Macro
  4. GridSearchCV on C        — find best regularisation strength

Data source: Churn_Customer/dataset/E Commerce Dataset.xlsx (Sheet: E Comm)
Target: Churn (1 = churned, 0 = retained)

Validation: StratifiedKFold (5 splits, shuffle=True, random_state=42)
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    make_scorer,
    precision_recall_curve,
)


# =========================================================
# 1. Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "dataset" / "E Commerce Dataset.xlsx"

print("=" * 70)
print("Churn Customer — Logistic Regression Experiment 2 (Improved)")
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
print(f"\nLabel distribution: Churn=0 -> {n0:,} ({n0/len(df)*100:.1f}%)  |  Churn=1 -> {n1:,} ({n1/len(df)*100:.1f}%)")


# =========================================================
# 4. Feature engineering
# =========================================================

print("\nApplying feature engineering...")

# Safe division helper
def safe_div(a, b, fill=0.0):
    return np.where(b > 0, a / b, fill)

# 4a. AvgCashbackPerOrder — how much cashback per order on average
df["AvgCashbackPerOrder"] = safe_div(
    df["CashbackAmount"].fillna(0),
    df["OrderCount"].fillna(0),
)

# 4b. IsHighComplainer — binary flag: customer filed a complaint
df["IsHighComplainer"] = (df["Complain"].fillna(0) == 1).astype(int)

# 4c. LowSatisfaction — satisfaction score of 1 or 2
df["LowSatisfaction"] = (df["SatisfactionScore"].fillna(3) <= 2).astype(int)

# 4d. DaysSinceOrderBucket — recency bucket (0-7=recent, 8-30=medium, 31+=long)
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

DROP_COLS = ["CustomerID", label_col]
DROP_COLS = [c for c in DROP_COLS if c in df.columns]

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

print(f"\nNumeric features  ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")


# =========================================================
# 6. Pipeline builder
# =========================================================

def build_pipeline(C=1.0):
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
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        )),
    ])


# =========================================================
# 7. GridSearchCV — find best C
# =========================================================

print("\n" + "=" * 70)
print("Step 1: GridSearchCV — finding best C (regularisation strength)...")
print("=" * 70)

inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    build_pipeline(),
    param_grid={"classifier__C": [0.01, 0.1, 1, 10, 100]},
    scoring="roc_auc",
    cv=inner_cv,
    n_jobs=1,
    verbose=0,
)
grid_search.fit(X, y)

best_C = grid_search.best_params_["classifier__C"]
print(f"\nBest C: {best_C}  (ROC AUC = {grid_search.best_score_:.4f})")

print("\nAll C results:")
for mean, std, params in zip(
    grid_search.cv_results_["mean_test_score"],
    grid_search.cv_results_["std_test_score"],
    grid_search.cv_results_["params"],
):
    print(f"  C={params['classifier__C']:>6} -> ROC AUC {mean:.4f} +/- {std:.4f}")


# =========================================================
# 8. Outer 5-Fold CV with best C
# =========================================================

print("\n" + "=" * 70)
print(f"Step 2: 5-Fold CV with class_weight='balanced', C={best_C}...")
print("=" * 70)

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = build_pipeline(C=best_C)

y_pred_oof = cross_val_predict(model, X, y, cv=outer_cv, method="predict", n_jobs=1)
y_proba_oof = cross_val_predict(model, X, y, cv=outer_cv, method="predict_proba", n_jobs=1)[:, 1]

print("\nOut-of-fold metrics (default threshold = 0.5):")
print(f"  Accuracy:          {accuracy_score(y, y_pred_oof):.4f}")
print(f"  Balanced Accuracy: {accuracy_score(y, y_pred_oof):.4f}")
print(f"  Precision (Ch=1):  {precision_score(y, y_pred_oof, pos_label=1, zero_division=0):.4f}")
print(f"  Recall    (Ch=1):  {recall_score(y, y_pred_oof, pos_label=1, zero_division=0):.4f}")
print(f"  F1        (Ch=1):  {f1_score(y, y_pred_oof, pos_label=1, zero_division=0):.4f}")
print(f"  F1 Macro:          {f1_score(y, y_pred_oof, average='macro', zero_division=0):.4f}")
print(f"  ROC AUC:           {roc_auc_score(y, y_proba_oof):.4f}")

print("\nConfusion Matrix (threshold=0.5):")
print(confusion_matrix(y, y_pred_oof))


# =========================================================
# 9. Threshold tuning — maximise F1 Macro
# =========================================================

print("\n" + "=" * 70)
print("Step 3: Threshold tuning via Precision-Recall curve...")
print("=" * 70)

_, _, thresholds = precision_recall_curve(y, y_proba_oof)

best_threshold = 0.5
best_f1_macro = 0.0

for t in thresholds:
    y_pred_t = (y_proba_oof >= t).astype(int)
    f1_m = f1_score(y, y_pred_t, average="macro", zero_division=0)
    if f1_m > best_f1_macro:
        best_f1_macro = f1_m
        best_threshold = t

print(f"\nOptimal threshold: {best_threshold:.4f}  (F1 Macro = {best_f1_macro:.4f})")

y_pred_tuned = (y_proba_oof >= best_threshold).astype(int)

print("\nOut-of-fold metrics (optimal threshold):")
print(f"  Accuracy:          {accuracy_score(y, y_pred_tuned):.4f}")
print(f"  Balanced Accuracy: {accuracy_score(y, y_pred_tuned):.4f}")
print(f"  Precision (Ch=1):  {precision_score(y, y_pred_tuned, pos_label=1, zero_division=0):.4f}")
print(f"  Recall    (Ch=1):  {recall_score(y, y_pred_tuned, pos_label=1, zero_division=0):.4f}")
print(f"  F1        (Ch=1):  {f1_score(y, y_pred_tuned, pos_label=1, zero_division=0):.4f}")
print(f"  F1 Macro:          {f1_score(y, y_pred_tuned, average='macro', zero_division=0):.4f}")
print(f"  ROC AUC:           {roc_auc_score(y, y_proba_oof):.4f}")

print("\nConfusion Matrix (optimal threshold):")
print(confusion_matrix(y, y_pred_tuned))

print("\nClassification Report (optimal threshold):")
print(classification_report(y, y_pred_tuned, zero_division=0))


# =========================================================
# 10. Threshold sweep table
# =========================================================

print("\nThreshold sweep (Recall vs Precision tradeoff):")
print(f"{'Threshold':>10} {'Precision(1)':>13} {'Recall(1)':>10} {'F1 Macro':>10}")
for t in np.arange(0.1, 0.9, 0.05):
    y_t = (y_proba_oof >= t).astype(int)
    p1 = precision_score(y, y_t, pos_label=1, zero_division=0)
    r1 = recall_score(y, y_t, pos_label=1, zero_division=0)
    fm = f1_score(y, y_t, average="macro", zero_division=0)
    marker = " << optimal" if abs(t - best_threshold) < 0.025 else ""
    print(f"{t:>10.2f} {p1:>13.4f} {r1:>10.4f} {fm:>10.4f}{marker}")


# =========================================================
# 11. Save summary
# =========================================================

summary = {
    "best_C": best_C,
    "optimal_threshold": round(float(best_threshold), 4),
    # default threshold metrics
    "accuracy_default": round(accuracy_score(y, y_pred_oof), 6),
    "recall_1_default": round(recall_score(y, y_pred_oof, pos_label=1, zero_division=0), 6),
    "precision_1_default": round(precision_score(y, y_pred_oof, pos_label=1, zero_division=0), 6),
    "f1_1_default": round(f1_score(y, y_pred_oof, pos_label=1, zero_division=0), 6),
    "f1_macro_default": round(f1_score(y, y_pred_oof, average="macro", zero_division=0), 6),
    "roc_auc": round(roc_auc_score(y, y_proba_oof), 6),
    # optimal threshold metrics
    "accuracy_tuned": round(accuracy_score(y, y_pred_tuned), 6),
    "recall_1_tuned": round(recall_score(y, y_pred_tuned, pos_label=1, zero_division=0), 6),
    "precision_1_tuned": round(precision_score(y, y_pred_tuned, pos_label=1, zero_division=0), 6),
    "f1_1_tuned": round(f1_score(y, y_pred_tuned, pos_label=1, zero_division=0), 6),
    "f1_macro_tuned": round(f1_score(y, y_pred_tuned, average="macro", zero_division=0), 6),
}

SUMMARY_FILE = BASE_DIR / "logistic_experiment_2_summary.csv"
pd.DataFrame([summary], index=["logistic_experiment_2"]).to_csv(SUMMARY_FILE, index=True)
print(f"\nSummary saved: {SUMMARY_FILE}")

print("\n" + "=" * 70)
print("Experiment 2 complete.")
print("=" * 70)
