"""
Churn Customer — Logistic Regression Experiment 1

Baseline model: simplest possible pipeline, no feature engineering,
no threshold tuning, no class weighting.

Data source: Churn_Customer/dataset/E Commerce Dataset.xlsx (Sheet: E Comm)
Target: Churn (1 = churned, 0 = retained)

Validation: StratifiedKFold (5 splits, shuffle=True, random_state=42)
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    make_scorer,
)


# =========================================================
# 1. Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "dataset" / "E Commerce Dataset.xlsx"

print("=" * 70)
print("Churn Customer — Logistic Regression Experiment 1 (Baseline)")
print("=" * 70)

print(f"\nLooking for dataset: {DATA_FILE}")

if not DATA_FILE.exists():
    dataset_dir = BASE_DIR.parent / "dataset"
    print("\nDataset file not found.")
    print(f"Expected path: {DATA_FILE}")
    if dataset_dir.exists():
        files = list(dataset_dir.iterdir())
        if files:
            print(f"\nFiles found in {dataset_dir}:")
            for f in files:
                print(f"  {f.name}")
        else:
            print(f"\nDataset folder exists but is empty.")
    else:
        print(f"\nDataset folder does not exist: {dataset_dir}")
    print("\nPlease download the dataset from Kaggle and place it at:")
    print(f"  {DATA_FILE}")
    print("See README_data_extraction.md for detailed instructions.")
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")


# =========================================================
# 2. Load data
# =========================================================

df = pd.read_excel(DATA_FILE, sheet_name="E Comm")
df.columns = df.columns.str.strip()

print(f"\nDataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Columns: {df.columns.tolist()}")


# =========================================================
# 3. Validate label
# =========================================================

label_col = "Churn"

if label_col not in df.columns:
    raise ValueError(
        f"Label column '{label_col}' not found. Available: {df.columns.tolist()}"
    )

df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
missing_labels = df[label_col].isna().sum()
if missing_labels > 0:
    print(f"\nWarning: {missing_labels} rows with missing label — dropping them.")
    df = df.dropna(subset=[label_col])

df[label_col] = df[label_col].astype(int)

n0 = (df[label_col] == 0).sum()
n1 = (df[label_col] == 1).sum()
print(f"\nLabel distribution:")
print(f"  Churn = 0 (retained): {n0:,} ({n0/len(df)*100:.1f}%)")
print(f"  Churn = 1 (churned):  {n1:,} ({n1/len(df)*100:.1f}%)")
print(f"  Imbalance ratio: {n0/n1:.2f}:1")


# =========================================================
# 4. Feature selection
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
]
categorical_features = [
    "PreferredLoginDevice", "PreferredPaymentMode", "Gender",
    "PreferedOrderCat", "MaritalStatus",
]

# Keep only columns that exist in the data
numeric_features = [c for c in numeric_features if c in X.columns]
categorical_features = [c for c in categorical_features if c in X.columns]

print(f"\nNumeric features ({len(numeric_features)}): {numeric_features}")
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
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )),
])


# =========================================================
# 6. Cross Validation
# =========================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision_0": make_scorer(precision_score, pos_label=0, zero_division=0),
    "recall_0": make_scorer(recall_score, pos_label=0, zero_division=0),
    "f1_0": make_scorer(f1_score, pos_label=0, zero_division=0),
    "precision_1": make_scorer(precision_score, pos_label=1, zero_division=0),
    "recall_1": make_scorer(recall_score, pos_label=1, zero_division=0),
    "f1_1": make_scorer(f1_score, pos_label=1, zero_division=0),
    "f1_macro": "f1_macro",
    "roc_auc": "roc_auc",
}

print("\n" + "=" * 70)
print("Running 5-Fold Stratified Cross Validation...")
print("=" * 70)

cv_results = cross_validate(
    model, X, y,
    cv=cv, scoring=scoring, n_jobs=1, return_train_score=False,
)

print("\nCross Validation results (5-Fold):")
for metric_name in scoring.keys():
    values = cv_results[f"test_{metric_name}"]
    print(f"  {metric_name:25s}: {values.mean():.4f} ± {values.std():.4f}")


# =========================================================
# 7. Aggregated out-of-fold metrics
# =========================================================

y_pred_oof = cross_val_predict(model, X, y, cv=cv, method="predict", n_jobs=1)
y_proba_oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]

print("\nAggregated out-of-fold metrics:")
print(f"  Accuracy:          {accuracy_score(y, y_pred_oof):.4f}")
print(f"  Balanced Accuracy: {balanced_accuracy_score(y, y_pred_oof):.4f}")
print(f"  Precision class 0: {precision_score(y, y_pred_oof, pos_label=0, zero_division=0):.4f}")
print(f"  Recall class 0:    {recall_score(y, y_pred_oof, pos_label=0, zero_division=0):.4f}")
print(f"  F1 class 0:        {f1_score(y, y_pred_oof, pos_label=0, zero_division=0):.4f}")
print(f"  Precision class 1: {precision_score(y, y_pred_oof, pos_label=1, zero_division=0):.4f}")
print(f"  Recall class 1:    {recall_score(y, y_pred_oof, pos_label=1, zero_division=0):.4f}")
print(f"  F1 class 1:        {f1_score(y, y_pred_oof, pos_label=1, zero_division=0):.4f}")
print(f"  F1 Macro:          {f1_score(y, y_pred_oof, average='macro', zero_division=0):.4f}")
print(f"  ROC AUC:           {roc_auc_score(y, y_proba_oof):.4f}")

print("\nAggregated Confusion Matrix:")
print(confusion_matrix(y, y_pred_oof))

print("\nClassification Report:")
print(classification_report(y, y_pred_oof, zero_division=0))


# =========================================================
# 8. Save CV summary
# =========================================================

summary = {}
for metric_name in scoring.keys():
    values = cv_results[f"test_{metric_name}"]
    summary[f"{metric_name}_mean"] = round(values.mean(), 6)
    summary[f"{metric_name}_std"] = round(values.std(), 6)

summary["oof_accuracy"] = round(accuracy_score(y, y_pred_oof), 6)
summary["oof_recall_0"] = round(recall_score(y, y_pred_oof, pos_label=0, zero_division=0), 6)
summary["oof_recall_1"] = round(recall_score(y, y_pred_oof, pos_label=1, zero_division=0), 6)
summary["oof_f1_macro"] = round(f1_score(y, y_pred_oof, average="macro", zero_division=0), 6)
summary["oof_roc_auc"] = round(roc_auc_score(y, y_proba_oof), 6)

SUMMARY_FILE = BASE_DIR / "logistic_experiment_1_summary.csv"
pd.DataFrame([summary], index=["logistic_experiment_1"]).to_csv(SUMMARY_FILE, index=True)
print(f"\nSummary saved: {SUMMARY_FILE}")

print("\n" + "=" * 70)
print("Experiment 1 complete.")
print("=" * 70)
