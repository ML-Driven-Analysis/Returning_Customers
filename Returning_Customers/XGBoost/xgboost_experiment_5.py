"""
Experiment 5 — Minimal Business Features

Goal: Test whether reducing to a clean, pre-purchase feature set improves
      XGBoost performance compared to the full engineered feature set.

Features kept: Age, Gender, City, Product_Category, Unit_Price, Quantity,
               Discount_Amount, Total_Amount, Payment_Method, Device_Type,
               Session_Duration_Minutes, Pages_Viewed, Month, DayOfWeek

Features removed: post-purchase (Delivery_Time_Days, Customer_Rating),
                  redundant binary flags, over-engineered ratios, identifiers.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    make_scorer,
)
from xgboost import XGBClassifier


# =========================================================
# 1. Load dataset
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "dataset" / "ecommerce_customer_behavior_dataset.csv"

print("Running script from:")
print(BASE_DIR)
print("\nLoading dataset from:")
print(DATA_FILE)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

print("\nDataset loaded successfully.")
print("Shape:", df.shape)


# =========================================================
# 2. Validate required columns
# =========================================================

label_col = "Is_Returning_Customer"

required_columns = [
    "Age", "Gender", "City", "Product_Category",
    "Unit_Price", "Quantity", "Discount_Amount", "Total_Amount",
    "Payment_Method", "Device_Type",
    "Session_Duration_Minutes", "Pages_Viewed",
    "Date", label_col,
]

missing = [c for c in required_columns if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

print("\nAll required columns present.")


# =========================================================
# 3. Convert label to 0 / 1
# =========================================================

def convert_label(value):
    s = str(value).strip().lower()
    if s in ["true", "1", "yes", "returning", "returning customer"]:
        return 1
    if s in ["false", "0", "no", "not returning", "not returning customer"]:
        return 0
    raise ValueError(f"Unknown label value: {value!r}")


df[label_col] = df[label_col].apply(convert_label)

n0 = (df[label_col] == 0).sum()
n1 = (df[label_col] == 1).sum()
print(f"\nLabel distribution: class 0={n0}, class 1={n1}")
print(f"scale_pos_weight = {n0}/{n1} = {n0/n1:.4f}")


# =========================================================
# 4. Drop identifier columns
# =========================================================

id_cols = ["Order_ID", "Customer_ID", "Serial_Number", "serial_number",
           "Serial", "serial", "Index", "index", "Unnamed: 0"]
id_cols = [c for c in id_cols if c in df.columns]
df = df.drop(columns=id_cols)


# =========================================================
# 5. Extract Month and DayOfWeek from Date, then drop Date
# =========================================================

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek
df = df.drop(columns=["Date"])


# =========================================================
# 6. Keep only the minimal feature set
# =========================================================

MINIMAL_FEATURES = [
    "Age", "Gender", "City", "Product_Category",
    "Unit_Price", "Quantity", "Discount_Amount", "Total_Amount",
    "Payment_Method", "Device_Type",
    "Session_Duration_Minutes", "Pages_Viewed",
    "Month", "DayOfWeek",
]

# Confirm all minimal features exist after preprocessing
missing_minimal = [f for f in MINIMAL_FEATURES if f not in df.columns]
if missing_minimal:
    raise ValueError(f"Minimal features missing after preprocessing: {missing_minimal}")

X = df[MINIMAL_FEATURES].copy()
y = df[label_col].copy()

print(f"\nFeature set size: {X.shape[1]} features")
print("Features:", MINIMAL_FEATURES)

# Force numeric types for numeric columns
numeric_features = [
    "Age", "Unit_Price", "Quantity", "Discount_Amount", "Total_Amount",
    "Session_Duration_Minutes", "Pages_Viewed", "Month", "DayOfWeek",
]
categorical_features = [
    "Gender", "City", "Product_Category", "Payment_Method", "Device_Type",
]

for col in numeric_features:
    X[col] = pd.to_numeric(X[col], errors="coerce")

print(f"\nNumeric features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")


# =========================================================
# 7. Build pipeline
# =========================================================

numeric_transformer = SimpleImputer(strategy="median")

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# Best params from Experiment 3 (RandomizedSearchCV)
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=1,
        gamma=0.5,
        scale_pos_weight=0.6722,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )),
])


# =========================================================
# 8. Cross Validation setup
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


# =========================================================
# 9. Run experiment
# =========================================================

print("\n" + "=" * 70)
print("XGBoost Experiment 5 — Minimal Business Features")
print("=" * 70)

cv_results = cross_validate(
    model, X, y,
    cv=cv, scoring=scoring, n_jobs=1, return_train_score=False,
)

print("\nCross Validation results (5-Fold):")
for metric_name in scoring.keys():
    values = cv_results[f"test_{metric_name}"]
    print(f"  {metric_name}: {values.mean():.4f} ± {values.std():.4f}")

# Out-of-fold aggregated predictions
y_pred_oof = cross_val_predict(model, X, y, cv=cv, method="predict", n_jobs=1)
y_proba_oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]

print("\nAggregated out-of-fold metrics:")
print(f"  Accuracy:          {accuracy_score(y, y_pred_oof):.4f}")
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
# 10. Feature importance
# =========================================================

print("=" * 70)
print("Feature Importance — Top 15")
print("=" * 70)

try:
    model.fit(X, y)
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False)

    print(importance_df.head(15).to_string(index=False))

    IMPORTANCE_FILE = BASE_DIR / "xgboost_experiment_5_feature_importance.csv"
    importance_df.to_csv(IMPORTANCE_FILE, index=False)
    print(f"\nFeature importance saved: {IMPORTANCE_FILE}")

except Exception as e:
    print(f"\nCould not compute feature importance: {e}")


# =========================================================
# 11. Save CV summary
# =========================================================

summary = {}
for metric_name in scoring.keys():
    values = cv_results[f"test_{metric_name}"]
    summary[f"{metric_name}_mean"] = round(values.mean(), 6)
    summary[f"{metric_name}_std"] = round(values.std(), 6)

# Append OOF aggregated metrics
summary["oof_accuracy"] = round(accuracy_score(y, y_pred_oof), 6)
summary["oof_recall_0"] = round(recall_score(y, y_pred_oof, pos_label=0, zero_division=0), 6)
summary["oof_f1_macro"] = round(f1_score(y, y_pred_oof, average="macro", zero_division=0), 6)
summary["oof_roc_auc"] = round(roc_auc_score(y, y_proba_oof), 6)

SUMMARY_FILE = BASE_DIR / "xgboost_experiment_5_summary.csv"
pd.DataFrame([summary], index=["xgboost_experiment_5"]).to_csv(SUMMARY_FILE, index=True)
print(f"\nSummary saved: {SUMMARY_FILE}")

print("\n" + "=" * 70)
print("Experiment 5 complete.")
print("=" * 70)
