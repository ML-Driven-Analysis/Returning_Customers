"""
Churn Customer — Logistic Regression Experiment 1 (with held-out test set)

Same baseline pipeline as logistic_experiment_1.py, but with a proper
train/test split: 70% train, 30% test (stratified, random_state=42).

The test set is held out completely — it is NOT used in cross-validation
or any model selection. It is only touched once, at the very end, to
report a final unbiased estimate of generalisation performance.

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
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation_utils import evaluate_model_cv, evaluate_predictions, print_evaluation


# =========================================================
# 1. Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "dataset" / "E Commerce Dataset.xlsx"

print("=" * 70)
print("Churn Customer - Logistic Regression Experiment 1 (Baseline + Holdout)")
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
# 3. Validate label
# =========================================================

label_col = "Churn"
df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
df = df.dropna(subset=[label_col])
df[label_col] = df[label_col].astype(int)

n0 = (df[label_col] == 0).sum()
n1 = (df[label_col] == 1).sum()
print(f"\nLabel distribution (full dataset):")
print(f"  Churn = 0 (retained): {n0:,} ({n0/len(df)*100:.1f}%)")
print(f"  Churn = 1 (churned):  {n1:,} ({n1/len(df)*100:.1f}%)")
print(f"  Imbalance ratio: {n0/n1:.2f}:1")


# =========================================================
# 4. Feature selection
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

print(f"\nNumeric features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")


# =========================================================
# 5. Train / Test split (held-out test - touched only at the end)
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
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )),
])


# =========================================================
# 7. Cross-validation evaluation - TRAIN set only
# =========================================================

print("\n" + "=" * 70)
print("Running 5-Fold Stratified Cross-Validation on TRAIN set...")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_results = evaluate_model_cv(model, X_train, y_train, cv, thresholds=[0.5])
print_evaluation(train_results, label="Train CV (5-Fold, threshold=0.5)")


# =========================================================
# 8. Final holdout check - TEST set, touched exactly once
# =========================================================

print("\n" + "=" * 70)
print("Final holdout evaluation on TEST set (model fit on full TRAIN set)")
print("=" * 70)

model.fit(X_train, y_train)
y_proba_test = model.predict_proba(X_test)[:, 1]

test_results = evaluate_predictions(y_test, y_proba_test, thresholds=[0.5])
print_evaluation(test_results, label="Held-out TEST (threshold=0.5)")


# =========================================================
# 9. Save summary
# =========================================================

m_train = train_results["thresholds"][0.5]
m_test = test_results["thresholds"][0.5]

summary = {
    "train_accuracy": round(m_train["accuracy"], 6),
    "train_balanced_accuracy": round(m_train["balanced_accuracy"], 6),
    "train_precision_1": round(m_train["precision_1"], 6),
    "train_recall_1": round(m_train["recall_1"], 6),
    "train_f1_1": round(m_train["f1_1"], 6),
    "train_f1_macro": round(m_train["f1_macro"], 6),
    "train_roc_auc": round(m_train["roc_auc"], 6),
    "test_accuracy": round(m_test["accuracy"], 6),
    "test_balanced_accuracy": round(m_test["balanced_accuracy"], 6),
    "test_precision_1": round(m_test["precision_1"], 6),
    "test_recall_1": round(m_test["recall_1"], 6),
    "test_f1_1": round(m_test["f1_1"], 6),
    "test_f1_macro": round(m_test["f1_macro"], 6),
    "test_roc_auc": round(m_test["roc_auc"], 6),
}

SUMMARY_FILE = BASE_DIR / "logistic_experiment_1_holdout_summary.csv"
pd.DataFrame([summary], index=["logistic_experiment_1_holdout"]).to_csv(SUMMARY_FILE, index=True)
print(f"\nSummary saved: {SUMMARY_FILE}")

print("\n" + "=" * 70)
print("Experiment 1 (holdout) complete.")
print("=" * 70)
