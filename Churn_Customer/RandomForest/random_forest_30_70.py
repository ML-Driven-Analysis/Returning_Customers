# """
# Churn Customer — Random Forest Experiment 1 (70/30 Split — Focused Output)

# Baseline model: simplest possible pipeline, no feature engineering,
# no threshold tuning. Incorporates balanced class weights for tree building.

# Data source: Churn_Customer/dataset/E Commerce Dataset.xlsx (Sheet: E Comm)
# Target: Churn (1 = churned, 0 = retained)

# Validation: Stratified 70/30 Train/Test Holdout Split
# Output: Focused on Class 1 (Churned) performance metrics and Overall Accuracy.
# """

# import pandas as pd
# import numpy as np
# from pathlib import Path

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import (
#     accuracy_score,
#     balanced_accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score,
#     confusion_matrix,
#     classification_report,
# )


# # =========================================================
# # 1. Paths
# # =========================================================

# BASE_DIR = Path(__file__).resolve().parent
# DATA_FILE = BASE_DIR.parent / "dataset" / "E Commerce Dataset.xlsx"

# print("=" * 70)
# print("Churn Customer — Random Forest Experiment 1 (70/30 Holdout)")
# print("=" * 70)

# print(f"\nLooking for dataset: {DATA_FILE}")

# if not DATA_FILE.exists():
#     dataset_dir = BASE_DIR.parent / "dataset"
#     print("\nDataset file not found.")
#     print(f"Expected path: {DATA_FILE}")
#     if dataset_dir.exists():
#         files = list(dataset_dir.iterdir())
#         if files:
#             print(f"\nFiles found in {dataset_dir}:")
#             for f in files:
#                 print(f"  {f.name}")
#         else:
#             print(f"\nDataset folder exists but is empty.")
#     else:
#         print(f"\nDataset folder does not exist: {dataset_dir}")
#     print("\nPlease download the dataset from Kaggle and place it at:")
#     print(f"  {DATA_FILE}")
#     print("See README_data_extraction.md for detailed instructions.")
#     raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")


# # =========================================================
# # 2. Load data
# # =========================================================

# df = pd.read_excel(DATA_FILE, sheet_name="E Comm")
# df.columns = df.columns.str.strip()

# print(f"\nDataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
# print(f"Columns: {df.columns.tolist()}")


# # =========================================================
# # 3. Validate label
# # =========================================================

# label_col = "Churn"

# if label_col not in df.columns:
#     raise ValueError(
#         f"Label column '{label_col}' not found. Available: {df.columns.tolist()}"
#     )

# df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
# missing_labels = df[label_col].isna().sum()
# if missing_labels > 0:
#     print(f"\nWarning: {missing_labels} rows with missing label — dropping them.")
#     df = df.dropna(subset=[label_col])

# df[label_col] = df[label_col].astype(int)

# n0 = (df[label_col] == 0).sum()
# n1 = (df[label_col] == 1).sum()
# print(f"\nLabel distribution:")
# print(f"  Churn = 0 (retained): {n0:,} ({n0/len(df)*100:.1f}%)")
# print(f"  Churn = 1 (churned):  {n1:,} ({n1/len(df)*100:.1f}%)")
# print(f"  Imbalance ratio: {n0/n1:.2f}:1")


# # =========================================================
# # 4. Feature selection & Data Type Protection
# # =========================================================

# DROP_COLS = ["CustomerID", label_col]
# DROP_COLS = [c for c in DROP_COLS if c in df.columns]

# X = df.drop(columns=DROP_COLS)
# y = df[label_col]

# numeric_features = [
#     "Tenure", "WarehouseToHome", "HourSpendOnApp", "NumberOfDeviceRegistered",
#     "SatisfactionScore", "NumberOfAddress", "Complain",
#     "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount",
#     "DaySinceLastOrder", "CashbackAmount", "CityTier",
# ]
# categorical_features = [
#     "PreferredLoginDevice", "PreferredPaymentMode", "Gender",
#     "PreferedOrderCat", "MaritalStatus",
# ]

# # Keep only columns that exist in the data
# numeric_features = [c for c in numeric_features if c in X.columns]
# categorical_features = [c for c in categorical_features if c in X.columns]

# # Data type protection: cast categorical features to clean string formats
# for col in categorical_features:
#     X[col] = X[col].astype(str)

# print(f"\nNumeric features ({len(numeric_features)}): {numeric_features}")
# print(f"Categorical features ({len(categorical_features)}): {categorical_features}")


# # =========================================================
# # 5. Pipeline
# # =========================================================

# numeric_transformer = Pipeline(steps=[
#     ("imputer", SimpleImputer(strategy="median")),
# ])

# categorical_transformer = Pipeline(steps=[
#     ("imputer", SimpleImputer(strategy="most_frequent")),
#     ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
# ])

# preprocessor = ColumnTransformer(transformers=[
#     ("num", numeric_transformer, numeric_features),
#     ("cat", categorical_transformer, categorical_features),
# ])

# model = Pipeline(steps=[
#     ("preprocessor", preprocessor),
#     ("classifier", RandomForestClassifier(
#         n_estimators=100,        # Number of trees in the forest
#         max_depth=12,            # Depth limit to prevent rapid overfitting
#         class_weight="balanced", # Handles class imbalance inherently for churn data
#         random_state=42,         # Ensures full reproducibility across runs
#         n_jobs=-1                # Uses all available CPU cores for faster parallel execution
#     )),
# ])


# # =========================================================
# # 6. Train / Test split (70% Train, 30% Test Holdout)
# # =========================================================

# print("\n" + "=" * 70)
# print("Splitting dataset into 70% Train and 30% Test...")
# print("=" * 70)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, 
#     test_size=0.30, 
#     random_state=42, 
#     stratify=y
# )

# print(f"Train set shape: {X_train.shape[0]:,} rows")
# print(f"Test set shape:  {X_test.shape[0]:,} rows")


# # =========================================================
# # 7. Train model
# # =========================================================

# print("\nTraining Random Forest Model on 70% Train data...")
# model.fit(X_train, y_train)


# # =========================================================
# # 8. Predict & Evaluation on 30% Test Holdout (Class 1 + Accuracy)
# # =========================================================

# y_pred = model.predict(X_test)
# y_proba = model.predict_proba(X_test)[:, 1]

# # Calculate all scores (Keep these active so the CSV summary save works perfectly!)
# test_accuracy = accuracy_score(y_test, y_pred)
# test_balanced_acc = balanced_accuracy_score(y_test, y_pred)
# test_precision_0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
# test_recall_0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
# test_f1_0 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
# test_precision_1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
# test_recall_1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
# test_f1_1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
# test_f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
# test_roc_auc = roc_auc_score(y_test, y_proba)

# print("\n" + "=" * 70)
# print("Model Evaluation — Random Forest Results (Focused Output)")
# print("=" * 70)

# # Printing Accuracy AND Class 1 metrics to the terminal output
# print(f"  Overall Accuracy:            {test_accuracy:.4f}")
# print(f"  Precision class 1 (Churned): {test_precision_1:.4f}")
# print(f"  Recall class 1 (Churned):    {test_recall_1:.4f}")
# print(f"  F1 class 1 (Churned):        {test_f1_1:.4f}")
# print(f"  ROC AUC:                     {test_roc_auc:.4f}")

# # Extract and isolate row 1 from the classification report dictionary
# report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
# print("\nClassification Report (Class 1 Only):")
# print(f"              precision    recall  f1-score   support")
# print(f" Churned (1)       {report_dict['1']['precision']:.2f}      {report_dict['1']['recall']:.2f}      {report_dict['1']['f1-score']:.2f}      {int(report_dict['1']['support'])}")


# # =========================================================
# # 9. Save Summary Results
# # =========================================================

# summary = {
#     "accuracy_mean": round(test_accuracy, 6),
#     "accuracy_std": 0.0,  
#     "balanced_accuracy_mean": round(test_balanced_acc, 6),
#     "balanced_accuracy_std": 0.0,
#     "precision_0_mean": round(test_precision_0, 6),
#     "precision_0_std": 0.0,
#     "recall_0_mean": round(test_recall_0, 6),
#     "recall_0_std": 0.0,
#     "f1_0_mean": round(test_f1_0, 6),
#     "f1_0_std": 0.0,
#     "precision_1_mean": round(test_precision_1, 6),
#     "precision_1_std": 0.0,
#     "recall_1_mean": round(test_recall_1, 6),
#     "recall_1_std": 0.0,
#     "f1_macro_mean": round(test_f1_macro, 6),
#     "f1_macro_std": 0.0,
#     "roc_auc_mean": round(test_roc_auc, 6),
#     "roc_auc_std": 0.0,
#     "oof_accuracy": round(test_accuracy, 6),
#     "oof_recall_0": round(test_recall_0, 6),
#     "oof_recall_1": round(test_recall_1, 6),
#     "oof_f1_macro": round(test_f1_macro, 6),
#     "oof_roc_auc": round(test_roc_auc, 6),
# }

# SUMMARY_FILE = BASE_DIR / "random_forest_experiment_1_summary.csv"
# pd.DataFrame([summary], index=["random_forest_experiment_1"]).to_csv(SUMMARY_FILE, index=True)
# print(f"\nSummary saved: {SUMMARY_FILE}")

# print("\n" + "=" * 70)
# print("Random Forest 70/30 Split Holdout complete.")
# print("=" * 70)


"""
Churn Customer — Random Forest Experiment 1 (70/30 Split — Focused Output)

Baseline model: simplest possible pipeline, no feature engineering,
no threshold tuning. Incorporates balanced class weights for tree building.
Updated to dynamically compute and display the complete Confusion Matrix in text format.

Data source: Churn_Customer/dataset/E Commerce Dataset.xlsx (Sheet: E Comm)
Target: Churn (1 = churned, 0 = retained)

Validation: Stratified 70/30 Train/Test Holdout Split
Output: Focused on Class 1 (Churned) performance metrics, Overall Accuracy, and Confusion Matrix.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# =========================================================
# 1. Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "dataset" / "E Commerce Dataset.xlsx"

print("=" * 70)
print("Churn Customer — Random Forest Experiment 1 (70/30 Holdout)")
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
# 4. Feature selection & Data Type Protection
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

# Data type protection: cast categorical features to clean string formats
for col in categorical_features:
    X[col] = X[col].astype(str)

print(f"\nNumeric features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")


# =========================================================
# 5. Pipeline
# =========================================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
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
    ("classifier", RandomForestClassifier(
        n_estimators=100,        # Number of trees in the forest
        max_depth=12,            # Depth limit to prevent rapid overfitting
        class_weight="balanced", # Handles class imbalance inherently for churn data
        random_state=42,         # Ensures full reproducibility across runs
        n_jobs=-1                # Uses all available CPU cores for faster parallel execution
    )),
])


# =========================================================
# 6. Train / Test split (70% Train, 30% Test Holdout)
# =========================================================

print("\n" + "=" * 70)
print("Splitting dataset into 70% Train and 30% Test...")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.30, 
    random_state=42, 
    stratify=y
)

print(f"Train set shape: {X_train.shape[0]:,} rows")
print(f"Test set shape:  {X_test.shape[0]:,} rows")


# =========================================================
# 7. Train model
# =========================================================

print("\nTraining Random Forest Model on 70% Train data...")
model.fit(X_train, y_train)


# =========================================================
# 8. Predict & Evaluation on 30% Test Holdout (Class 1 + Accuracy)
# =========================================================

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate all scores 
test_accuracy = accuracy_score(y_test, y_pred)
test_balanced_acc = balanced_accuracy_score(y_test, y_pred)
test_precision_0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
test_recall_0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
test_f1_0 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
test_precision_1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
test_recall_1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
test_f1_1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
test_f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
test_roc_auc = roc_auc_score(y_test, y_proba)

print("\n" + "=" * 70)
print("Model Evaluation — Random Forest Results (Focused Output)")
print("=" * 70)

# Printing Accuracy AND Class 1 metrics to the terminal output
print(f"  Overall Accuracy:             {test_accuracy:.4f}")
print(f"  Precision class 1 (Churned): {test_precision_1:.4f}")
print(f"  Recall class 1 (Churned):    {test_recall_1:.4f}")
print(f"  F1 class 1 (Churned):        {test_f1_1:.4f}")
print(f"  ROC AUC:                     {test_roc_auc:.4f}")

# Extract and display the Confusion Matrix explicitly
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print("\nConfusion Matrix (Absolute Values):")
print(f"  -------------------------------------")
print(f"  |              | Predicted Retained | Predicted Churned  |")
print(f"  -------------------------------------")
print(f"  | Actual Stay  | TN: {tn:<14} | FP: {fp:<14} |")
print(f"  | Actual Churn | FN: {fn:<14} | TP: {tp:<14} |")
print(f"  -------------------------------------")

# Extract and isolate row 1 from the classification report dictionary
report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
print("\nClassification Report (Class 1 Only):")
print(f"              precision    recall  f1-score   support")
print(f" Churned (1)       {report_dict['1']['precision']:.2f}      {report_dict['1']['recall']:.2f}      {report_dict['1']['f1-score']:.2f}      {int(report_dict['1']['support'])}")


# =========================================================
# 9. Save Summary Results
# =========================================================

summary = {
    "accuracy_mean": round(test_accuracy, 6),
    "accuracy_std": 0.0,  
    "balanced_accuracy_mean": round(test_balanced_acc, 6),
    "balanced_accuracy_std": 0.0,
    "precision_0_mean": round(test_precision_0, 6),
    "precision_0_std": 0.0,
    "recall_0_mean": round(test_recall_0, 6),
    "recall_0_std": 0.0,
    "f1_0_mean": round(test_f1_0, 6),
    "f1_0_std": 0.0,
    "precision_1_mean": round(test_precision_1, 6),
    "precision_1_std": 0.0,
    "recall_1_mean": round(test_recall_1, 6),
    "recall_1_std": 0.0,
    "f1_macro_mean": round(test_f1_macro, 6),
    "f1_macro_std": 0.0,
    "roc_auc_mean": round(test_roc_auc, 6),
    "roc_auc_std": 0.0,
    "oof_accuracy": round(test_accuracy, 6),
    "oof_recall_0": round(test_recall_0, 6),
    "oof_recall_1": round(test_recall_1, 6),
    "oof_f1_macro": round(test_f1_macro, 6),
    "oof_roc_auc": round(test_roc_auc, 6),
}

SUMMARY_FILE = BASE_DIR / "random_forest_experiment_1_summary.csv"
pd.DataFrame([summary], index=["random_forest_experiment_1"]).to_csv(SUMMARY_FILE, index=True)
print(f"\nSummary saved: {SUMMARY_FILE}")

print("\n" + "=" * 70)
print("Random Forest 70/30 Split Holdout complete.")
print("=" * 70)