# """
# Churn Customer — Random Forest Experiment 1 (70/30 Split — Threshold Scan)

# Baseline model: simplest possible pipeline, no feature engineering.
# Incorporates balanced class weights and a dynamic threshold scan to optimize and maximize Recall.

# Data source: Churn_Customer/dataset/E Commerce Dataset.xlsx (Sheet: E Comm)
# Target: Churn (1 = churned, 0 = retained)

# Validation: Stratified 70/30 Train/Test Holdout Split
# Output: Comparative performance matrix across multiple decision thresholds.
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
# )


# # =========================================================
# # 1. Paths
# # =========================================================

# BASE_DIR = Path(__file__).resolve().parent
# DATA_FILE = BASE_DIR.parent / "dataset" / "E Commerce Dataset.xlsx"

# print("=" * 70)
# print("Churn Customer — Random Forest Experiment 1 (Threshold Optimization)")
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
# # 8. Dynamic Probability Threshold Scan (Isolate & Compare)
# # =========================================================

# # Extract predicted probabilities for Class 1 (Churned) from the test set
# y_proba = model.predict_proba(X_test)[:, 1]

# print("\n" + "=" * 85)
# print("DYNAMIC DECISION THRESHOLD SCAN — PERFORMANCE COMPARISON (CLASS 1 FOCUS)")
# print("=" * 85)
# print(f" {'Threshold':11s} | {'Accuracy':10s} | {'Precision (C1)':16s} | {'Recall (C1)':13s} | {'F1-Score (C1)':14s}")
# print("-" * 85)

# # Array of thresholds to scan from highly permissive (0.20) to conservative (0.60)
# thresholds_to_scan = [0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20]
# scan_records = []

# # Loop through each threshold dynamically to evaluate classification metrics
# for threshold in thresholds_to_scan:
#     preds = (y_proba >= threshold).astype(int)
    
#     acc = accuracy_score(y_test, preds)
#     b_acc = balanced_accuracy_score(y_test, preds)
#     p0 = precision_score(y_test, preds, pos_label=0, zero_division=0)
#     r0 = recall_score(y_test, preds, pos_label=0, zero_division=0)
#     f0 = f1_score(y_test, preds, pos_label=0, zero_division=0)
    
#     p1 = precision_score(y_test, preds, pos_label=1, zero_division=0)
#     r1 = recall_score(y_test, preds, pos_label=1, zero_division=0)
#     f1 = f1_score(y_test, preds, pos_label=1, zero_division=0)
    
#     f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)
#     roc_auc = roc_auc_score(y_test, y_proba)
    
#     # Print a clean row summarizing key class 1 outcomes for the terminal log
#     print(f"  {threshold:<10.2f} | {acc:<10.4f} | {p1:<16.4f} | {r1:<13.4f} | {f1:<14.4f}")
    
#     # Collect data for the global CSV summary storage structure
#     scan_records.append({
#         "threshold": threshold, "accuracy": acc, "balanced_accuracy": b_acc,
#         "precision_0": p0, "recall_0": r0, "f1_0": f0,
#         "precision_1": p1, "recall_1": r1, "f1_1": f1,
#         "f1_macro": f1_macro, "roc_auc": roc_auc
#     })

# print("-" * 85)

# # For final CSV compilation, we save the configuration matching our standard row structure using threshold 0.35
# # (This keeps the save pipeline active and structured seamlessly for your main experimental log file)
# chosen_idx = thresholds_to_scan.index(0.35)
# c_rec = scan_records[chosen_idx]


# # =========================================================
# # 9. Save Summary Results (Optimized configuration)
# # =========================================================

# summary = {
#     "accuracy_mean": round(c_rec["accuracy"], 6),
#     "accuracy_std": 0.0,  
#     "balanced_accuracy_mean": round(c_rec["balanced_accuracy"], 6),
#     "balanced_accuracy_std": 0.0,
#     "precision_0_mean": round(c_rec["precision_0"], 6),
#     "precision_0_std": 0.0,
#     "recall_0_mean": round(c_rec["recall_0"], 6),
#     "recall_0_std": 0.0,
#     "f1_0_mean": round(c_rec["f1_0"], 6),
#     "f1_0_std": 0.0,
#     "precision_1_mean": round(c_rec["precision_1"], 6),
#     "precision_1_std": 0.0,
#     "recall_1_mean": round(c_rec["recall_1"], 6),
#     "recall_1_std": 0.0,
#     "f1_macro_mean": round(c_rec["f1_macro"], 6),
#     "f1_macro_std": 0.0,
#     "roc_auc_mean": round(c_rec["roc_auc"], 6),
#     "roc_auc_std": 0.0,
#     "oof_accuracy": round(c_rec["accuracy"], 6),
#     "oof_recall_0": round(c_rec["recall_0"], 6),
#     "oof_recall_1": round(c_rec["recall_1"], 6),
#     "oof_f1_macro": round(c_rec["f1_macro"], 6),
#     "oof_roc_auc": round(c_rec["roc_auc"], 6),
# }

# SUMMARY_FILE = BASE_DIR / "random_forest_experiment_1_summary.csv"
# pd.DataFrame([summary], index=["random_forest_experiment_1"]).to_csv(SUMMARY_FILE, index=True)
# print(f"\nSummary configuration for Threshold 0.35 saved: {SUMMARY_FILE}")

# print("\n" + "=" * 70)
# print("Random Forest 70/30 Split Dynamic Threshold Scan complete.")
# print("=" * 70)



"""
Churn Customer — Random Forest Experiment 1 (70/30 Split — Threshold Scan)

Baseline model: simplest possible pipeline, no feature engineering.
Incorporates balanced class weights and a dynamic threshold scan to optimize and maximize Recall.
Added detailed Confusion Matrix (TN, FP, FN, TP) display for each scanned threshold.

Data source: Churn_Customer/dataset/E Commerce Dataset.xlsx (Sheet: E Comm)
Target: Churn (1 = churned, 0 = retained)

Validation: Stratified 70/30 Train/Test Holdout Split
Output: Comparative performance matrix across multiple decision thresholds.
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
)


# =========================================================
# 1. Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "dataset" / "E Commerce Dataset.xlsx"

print("=" * 70)
print("Churn Customer — Random Forest Experiment 1 (Threshold Optimization)")
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
# 8. Dynamic Probability Threshold Scan (Isolate & Compare)
# =========================================================

# Extract predicted probabilities for Class 1 (Churned) from the test set
y_proba = model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 135)
print("DYNAMIC DECISION THRESHOLD SCAN — PERFORMANCE & CONFUSION MATRIX COMPARISON (CLASS 1 FOCUS)")
print("=" * 135)
# כאן הוספנו את עמודת הכותרת הברורה: Balanced Acc
print(f" {'Threshold':11s} | {'Accuracy':10s} | {'Balanced Acc':12s} | {'Precision (C1)':16s} | {'Recall (C1)':13s} | {'F1-Score (C1)':14s} | {'Confusion Matrix (TN, FP, FN, TP)':35s}")
print("-" * 135)

# Array of thresholds to scan from highly permissive (0.20) to conservative (0.60)
thresholds_to_scan = [0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20]
scan_records = []

# Loop through each threshold dynamically to evaluate classification metrics
for threshold in thresholds_to_scan:
    preds = (y_proba >= threshold).astype(int)
    
    acc = accuracy_score(y_test, preds)
    b_acc = balanced_accuracy_score(y_test, preds) # חישוב הציון המאוזן
    p0 = precision_score(y_test, preds, pos_label=0, zero_division=0)
    r0 = recall_score(y_test, preds, pos_label=0, zero_division=0)
    f0 = f1_score(y_test, preds, pos_label=0, zero_division=0)
    
    p1 = precision_score(y_test, preds, pos_label=1, zero_division=0)
    r1 = recall_score(y_test, preds, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, preds, pos_label=1, zero_division=0)
    
    f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Extract the elements of the confusion matrix for this specific threshold
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    cm_str = f"TN: {tn:<4} FP: {fp:<3} FN: {fn:<3} TP: {tp:<4}"
    
    # כאן הזרקנו את המשתנה b_acc ישירות לתוך פקודת ההדפסה של השורה
    print(f"  {threshold:<10.2f} | {acc:<10.4f} | {b_acc:<12.4f} | {p1:<16.4f} | {r1:<13.4f} | {f1:<14.4f} | {cm_str}")
    
    # Collect data for the global CSV summary storage structure
    scan_records.append({
        "threshold": threshold, "accuracy": acc, "balanced_accuracy": b_acc,
        "precision_0": p0, "recall_0": r0, "f1_0": f0,
        "precision_1": p1, "recall_1": r1, "f1_1": f1,
        "f1_macro": f1_macro, "roc_auc": roc_auc
    })

print("-" * 135)


# =========================================================
# 9. Save Summary Results (Optimized configuration)
# =========================================================

summary = {
    "accuracy_mean": round(c_rec["accuracy"], 6),
    "accuracy_std": 0.0,  
    "balanced_accuracy_mean": round(c_rec["balanced_accuracy"], 6),
    "balanced_accuracy_std": 0.0,
    "precision_0_mean": round(c_rec["precision_0"], 6),
    "precision_0_std": 0.0,
    "recall_0_mean": round(c_rec["recall_0"], 6),
    "recall_0_std": 0.0,
    "f1_0_mean": round(c_rec["f1_0"], 6),
    "f1_0_std": 0.0,
    "precision_1_mean": round(c_rec["precision_1"], 6),
    "precision_1_std": 0.0,
    "recall_1_mean": round(c_rec["recall_1"], 6),
    "recall_1_std": 0.0,
    "f1_macro_mean": round(c_rec["f1_macro"], 6),
    "f1_macro_std": 0.0,
    "roc_auc_mean": round(c_rec["roc_auc"], 6),
    "roc_auc_std": 0.0,
    "oof_accuracy": round(c_rec["accuracy"], 6),
    "oof_recall_0": round(c_rec["recall_0"], 6),
    "oof_recall_1": round(c_rec["recall_1"], 6),
    "oof_f1_macro": round(c_rec["f1_macro"], 6),
    "oof_roc_auc": round(c_rec["roc_auc"], 6),
}

SUMMARY_FILE = BASE_DIR / "random_forest_experiment_1_summary.csv"
pd.DataFrame([summary], index=["random_forest_experiment_1"]).to_csv(SUMMARY_FILE, index=True)
print(f"\nSummary configuration for Threshold 0.35 saved: {SUMMARY_FILE}")

print("\n" + "=" * 70)
print("Random Forest 70/30 Split Dynamic Threshold Scan complete.")
print("=" * 70)