"""
Churn Customer — Logistic Regression Experiment 2 (with held-out test set)

Same improvements as logistic_experiment_2.py (class_weight='balanced',
feature engineering, GridSearchCV on C, threshold tuning), but with a
proper train/test split: 70% train, 30% test (stratified, random_state=42).

The test set is held out completely — GridSearchCV, the outer 5-fold CV,
and threshold tuning all run on TRAIN only. TEST is touched exactly once,
at the end, to report an unbiased final estimate.

Evaluation is delegated to evaluation_utils.py (evaluate_model_cv /
evaluate_predictions / print_evaluation) so every experiment is scored
the same way.

Data source: Churn_Customer/dataset/E Commerce Dataset.xlsx (Sheet: E Comm)
Target: Churn (1 = churned, 0 = retained)

Validation: StratifiedKFold (5 splits, shuffle=True, random_state=42) on TRAIN only
Final check: single evaluation on held-out TEST set
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation_utils import evaluate_model_cv, evaluate_predictions, print_evaluation


# =========================================================
# 1. Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "dataset" / "E Commerce Dataset.xlsx"

print("=" * 70)
print("Churn Customer - Logistic Regression Experiment 2 (Improved + Holdout)")
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
print(f"\nLabel distribution (full dataset): Churn=0 -> {n0:,} ({n0/len(df)*100:.1f}%)  |  Churn=1 -> {n1:,} ({n1/len(df)*100:.1f}%)")


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

print(f"\nNumeric features  ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")


# =========================================================
# 6. Train / Test split (held-out test - touched only at the end)
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42,
)

print("\n" + "=" * 70)
print("Train / Test split (70% / 30%, stratified)")
print("=" * 70)
print(f"  Train: {X_train.shape[0]:,} rows  "
      f"(Churn=1: {(y_train == 1).sum():,} | {(y_train == 1).mean()*100:.1f}%)")
print(f"  Test:  {X_test.shape[0]:,} rows  "
      f"(Churn=1: {(y_test == 1).sum():,} | {(y_test == 1).mean()*100:.1f}%)")
print("\n  NOTE: X_test / y_test are not touched again until the final holdout check.")


# =========================================================
# 7. Pipeline builder
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
# 8. GridSearchCV - find best C, TRAIN only
# =========================================================

print("\n" + "=" * 70)
print("Step 1: GridSearchCV on TRAIN - finding best C (regularisation strength)...")
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
grid_search.fit(X_train, y_train)

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
# 9. Outer 5-Fold CV with best C - TRAIN only
# =========================================================

print("\n" + "=" * 70)
print(f"Step 2: 5-Fold CV on TRAIN with class_weight='balanced', C={best_C}...")
print("=" * 70)

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = build_pipeline(C=best_C)

train_results = evaluate_model_cv(model, X_train, y_train, outer_cv, thresholds=[0.5])
print_evaluation(train_results, label="Train CV (5-Fold, threshold=0.5)")

y_proba_train_oof = train_results["y_proba"]


# =========================================================
# 10. Threshold tuning - maximise F1 Macro (on TRAIN OOF only)
# =========================================================

print("\n" + "=" * 70)
print("Step 3: Threshold tuning via Precision-Recall curve (TRAIN OOF)...")
print("=" * 70)

_, _, thresholds = precision_recall_curve(y_train, y_proba_train_oof)

best_threshold = 0.5
best_f1_macro = 0.0

for t in thresholds:
    y_pred_t = (y_proba_train_oof >= t).astype(int)
    f1_m = f1_score(y_train, y_pred_t, average="macro", zero_division=0)
    if f1_m > best_f1_macro:
        best_f1_macro = f1_m
        best_threshold = t

print(f"\nOptimal threshold: {best_threshold:.4f}  (Train F1 Macro = {best_f1_macro:.4f})")

train_results_tuned = evaluate_model_cv(model, X_train, y_train, outer_cv, thresholds=[best_threshold])
print_evaluation(train_results_tuned, label=f"Train CV (5-Fold, threshold={best_threshold:.4f})")


# =========================================================
# 11. Final holdout check - TEST set, touched exactly once
# =========================================================

print("\n" + "=" * 70)
print("Final holdout evaluation on TEST set (model fit on full TRAIN set)")
print("=" * 70)

model.fit(X_train, y_train)
y_proba_test = model.predict_proba(X_test)[:, 1]

test_results = evaluate_predictions(y_test, y_proba_test, thresholds=[0.5, best_threshold])
print_evaluation(test_results, label="Held-out TEST")


# =========================================================
# 12. Save summary
# =========================================================

m_train_default = train_results["thresholds"][0.5]
m_train_tuned = train_results_tuned["thresholds"][best_threshold]
m_test_default = test_results["thresholds"][0.5]
m_test_tuned = test_results["thresholds"][best_threshold]

summary = {
    "best_C": best_C,
    "optimal_threshold": round(float(best_threshold), 4),
    # train, default threshold
    "train_accuracy_default": round(m_train_default["accuracy"], 6),
    "train_recall_1_default": round(m_train_default["recall_1"], 6),
    "train_precision_1_default": round(m_train_default["precision_1"], 6),
    "train_f1_1_default": round(m_train_default["f1_1"], 6),
    "train_f1_macro_default": round(m_train_default["f1_macro"], 6),
    "train_roc_auc": round(m_train_default["roc_auc"], 6),
    # train, optimal threshold
    "train_accuracy_tuned": round(m_train_tuned["accuracy"], 6),
    "train_recall_1_tuned": round(m_train_tuned["recall_1"], 6),
    "train_precision_1_tuned": round(m_train_tuned["precision_1"], 6),
    "train_f1_1_tuned": round(m_train_tuned["f1_1"], 6),
    "train_f1_macro_tuned": round(m_train_tuned["f1_macro"], 6),
    # test, default threshold
    "test_accuracy_default": round(m_test_default["accuracy"], 6),
    "test_recall_1_default": round(m_test_default["recall_1"], 6),
    "test_precision_1_default": round(m_test_default["precision_1"], 6),
    "test_f1_1_default": round(m_test_default["f1_1"], 6),
    "test_f1_macro_default": round(m_test_default["f1_macro"], 6),
    "test_roc_auc": round(m_test_default["roc_auc"], 6),
    # test, optimal threshold
    "test_accuracy_tuned": round(m_test_tuned["accuracy"], 6),
    "test_recall_1_tuned": round(m_test_tuned["recall_1"], 6),
    "test_precision_1_tuned": round(m_test_tuned["precision_1"], 6),
    "test_f1_1_tuned": round(m_test_tuned["f1_1"], 6),
    "test_f1_macro_tuned": round(m_test_tuned["f1_macro"], 6),
}

SUMMARY_FILE = BASE_DIR / "logistic_experiment_2_holdout_summary.csv"
pd.DataFrame([summary], index=["logistic_experiment_2_holdout"]).to_csv(SUMMARY_FILE, index=True)
print(f"\nSummary saved: {SUMMARY_FILE}")

print("\n" + "=" * 70)
print("Experiment 2 (holdout) complete.")
print("=" * 70)
