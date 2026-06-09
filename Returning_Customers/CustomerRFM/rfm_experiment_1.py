"""
CustomerRFM — Experiment 1: Customer-Level RFM Behavioral Profiling

Instead of predicting per-transaction (as all previous experiments did),
this experiment builds a behavioral profile per customer from their full
order history, then trains on those customer-level profiles.

Data sources:
  - for_test.csv (17,049 transactions, same 5,000 customers, avg 3.4 orders each)
    → used for feature engineering (RFM + behavioral aggregations)
  - dataset/ecommerce_customer_behavior_dataset.csv (5,000 rows, 1 per customer)
    → used for the ground-truth label (Is_Returning_Customer)

Result: 5,000 rows × ~25 customer-profile features → LightGBM classifier
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, make_scorer,
)
import lightgbm as lgb


# =========================================================
# 1. Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

MULTIORDER_FILE = ROOT_DIR / "for_test.csv"
ORIG_FILE = ROOT_DIR / "dataset" / "ecommerce_customer_behavior_dataset.csv"

print("=" * 70)
print("CustomerRFM — Experiment 1: Customer-Level Behavioral Profiling")
print("=" * 70)

for f in [MULTIORDER_FILE, ORIG_FILE]:
    if not f.exists():
        raise FileNotFoundError(f"Required file not found: {f}")

print(f"\nMulti-order file : {MULTIORDER_FILE}")
print(f"Labels file      : {ORIG_FILE}")


# =========================================================
# 2. Load data
# =========================================================

mo = pd.read_csv(MULTIORDER_FILE)
mo.columns = mo.columns.str.strip()

orig = pd.read_csv(ORIG_FILE)
orig.columns = orig.columns.str.strip()

print(f"\nMulti-order dataset : {mo.shape[0]:,} rows, {mo['Customer_ID'].nunique():,} unique customers")
print(f"Original dataset    : {orig.shape[0]:,} rows")


# =========================================================
# 3. Convert and validate label (from original dataset)
# =========================================================

def convert_label(value):
    s = str(value).strip().lower()
    if s in ["true", "1", "yes", "returning", "returning customer"]:
        return 1
    if s in ["false", "0", "no", "not returning", "not returning customer"]:
        return 0
    raise ValueError(f"Unknown label value: {value!r}")


orig["label"] = orig["Is_Returning_Customer"].apply(convert_label)
labels = orig[["Customer_ID", "label"]].copy()

n0 = (labels["label"] == 0).sum()
n1 = (labels["label"] == 1).sum()
print(f"\nLabel distribution (from original): class 0={n0}, class 1={n1}")
print(f"Imbalance ratio (class_weight scale): {n0/n1:.4f}")


# =========================================================
# 4. Prepare multi-order dataset
# =========================================================

mo["Date"] = pd.to_datetime(mo["Date"], errors="coerce")
mo = mo.sort_values(["Customer_ID", "Date"]).reset_index(drop=True)

# Order rank per customer (1 = first order)
mo["order_rank"] = mo.groupby("Customer_ID").cumcount() + 1

numeric_cols = [
    "Age", "Unit_Price", "Quantity", "Discount_Amount", "Total_Amount",
    "Session_Duration_Minutes", "Pages_Viewed", "Delivery_Time_Days", "Customer_Rating"
]
for col in numeric_cols:
    if col in mo.columns:
        mo[col] = pd.to_numeric(mo[col], errors="coerce")


# =========================================================
# 5. Build customer-level features (RFM + behavioral profile)
# =========================================================

print("\nBuilding customer-level RFM features...")

def mode_or_first(series):
    m = series.mode()
    return m.iloc[0] if len(m) > 0 else series.iloc[0]


# --- Frequency & Recency ---
freq_df = mo.groupby("Customer_ID").agg(
    n_orders=("Order_ID", "count"),
    first_order_date=("Date", "min"),
    last_order_date=("Date", "max"),
).reset_index()

freq_df["order_span_days"] = (
    freq_df["last_order_date"] - freq_df["first_order_date"]
).dt.days

# Inter-order gaps
gaps = (
    mo.groupby("Customer_ID")["Date"]
    .apply(lambda x: x.sort_values().diff().dt.days.dropna())
    .reset_index(level=0, drop=False)
    .rename(columns={"Date": "gap_days"})
)

if len(gaps) > 0 and "gap_days" in gaps.columns:
    gap_agg = gaps.groupby("Customer_ID")["gap_days"].agg(
        avg_gap_days="mean",
        min_gap_days="min",
        max_gap_days="max",
    ).reset_index()
else:
    gap_agg = pd.DataFrame(columns=["Customer_ID", "avg_gap_days", "min_gap_days", "max_gap_days"])

freq_df = freq_df.merge(gap_agg, on="Customer_ID", how="left")
freq_df[["avg_gap_days", "min_gap_days", "max_gap_days"]] = (
    freq_df[["avg_gap_days", "min_gap_days", "max_gap_days"]].fillna(0)
)

# --- Monetary ---
monetary_df = mo.groupby("Customer_ID").agg(
    total_spend=("Total_Amount", "sum"),
    avg_order_value=("Total_Amount", "mean"),
    max_order_value=("Total_Amount", "max"),
    std_order_value=("Total_Amount", "std"),
    total_discount=("Discount_Amount", "sum"),
    avg_discount=("Discount_Amount", "mean"),
    discount_usage_rate=("Discount_Amount", lambda x: (x > 0).mean()),
).reset_index()
monetary_df["std_order_value"] = monetary_df["std_order_value"].fillna(0)

# --- Product behavior ---
product_df = mo.groupby("Customer_ID").agg(
    n_unique_categories=("Product_Category", "nunique"),
    favorite_category=("Product_Category", mode_or_first),
    avg_unit_price=("Unit_Price", "mean"),
    avg_quantity=("Quantity", "mean"),
).reset_index()

# --- Session behavior ---
session_df = mo.groupby("Customer_ID").agg(
    avg_session_duration=("Session_Duration_Minutes", "mean"),
    std_session_duration=("Session_Duration_Minutes", "std"),
    avg_pages_viewed=("Pages_Viewed", "mean"),
).reset_index()
session_df["std_session_duration"] = session_df["std_session_duration"].fillna(0)

# --- Payment behavior ---
payment_df = mo.groupby("Customer_ID").agg(
    favorite_payment=("Payment_Method", mode_or_first),
    favorite_device=("Device_Type", mode_or_first),
).reset_index()

# --- Satisfaction ---
satisfaction_df = mo.groupby("Customer_ID").agg(
    avg_rating=("Customer_Rating", "mean"),
    min_rating=("Customer_Rating", "min"),
    avg_delivery_days=("Delivery_Time_Days", "mean"),
).reset_index()

# --- Demographics (from first order) ---
first_order = mo[mo["order_rank"] == 1][
    ["Customer_ID", "Age", "Gender", "City"]
].copy()

# --- Time patterns ---
mo["month"] = mo["Date"].dt.month
mo["dayofweek"] = mo["Date"].dt.dayofweek
time_df = mo.groupby("Customer_ID").agg(
    first_order_month=("month", "first"),
    most_common_dow=("dayofweek", mode_or_first),
).reset_index()


# =========================================================
# 6. Merge all features into one customer-level DataFrame
# =========================================================

customer_df = (
    freq_df[["Customer_ID", "n_orders", "order_span_days",
             "avg_gap_days", "min_gap_days", "max_gap_days"]]
    .merge(monetary_df, on="Customer_ID", how="left")
    .merge(product_df, on="Customer_ID", how="left")
    .merge(session_df, on="Customer_ID", how="left")
    .merge(payment_df, on="Customer_ID", how="left")
    .merge(satisfaction_df, on="Customer_ID", how="left")
    .merge(first_order, on="Customer_ID", how="left")
    .merge(time_df, on="Customer_ID", how="left")
    .merge(labels, on="Customer_ID", how="inner")
)

print(f"Customer profile DataFrame: {customer_df.shape[0]:,} rows × {customer_df.shape[1]} columns")

feature_cols = [c for c in customer_df.columns if c not in ["Customer_ID", "label"]]
print(f"Feature count: {len(feature_cols)}")
print("Features:", feature_cols)

X = customer_df[feature_cols].copy()
y = customer_df["label"].copy()

# Identify categorical features for LightGBM
categorical_features = ["favorite_category", "favorite_payment", "favorite_device", "Gender", "City"]
categorical_features = [c for c in categorical_features if c in X.columns]

# Convert categoricals to 'category' dtype so LightGBM handles them natively
for col in categorical_features:
    X[col] = X[col].astype("category")

print(f"\nNumeric features: {len([c for c in feature_cols if c not in categorical_features])}")
print(f"Categorical features: {categorical_features}")


# =========================================================
# 7. LightGBM model
# =========================================================

model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    num_leaves=15,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    class_weight="balanced",
    random_state=42,
    verbose=-1,
)


# =========================================================
# 8. Cross Validation
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
    print(f"  {metric_name:20s}: {values.mean():.4f} ± {values.std():.4f}")

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
# 9. Feature importance
# =========================================================

print("=" * 70)
print("Feature Importance — Top 20")
print("=" * 70)

try:
    model.fit(X, y)
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    print(importance_df.head(20).to_string(index=False))

    IMPORTANCE_FILE = BASE_DIR / "rfm_experiment_1_feature_importance.csv"
    importance_df.to_csv(IMPORTANCE_FILE, index=False)
    print(f"\nFeature importance saved: {IMPORTANCE_FILE}")

except Exception as e:
    print(f"\nCould not compute feature importance: {e}")


# =========================================================
# 10. Save summary
# =========================================================

summary = {}
for metric_name in scoring.keys():
    values = cv_results[f"test_{metric_name}"]
    summary[f"{metric_name}_mean"] = round(values.mean(), 6)
    summary[f"{metric_name}_std"] = round(values.std(), 6)

summary["oof_accuracy"] = round(accuracy_score(y, y_pred_oof), 6)
summary["oof_recall_0"] = round(recall_score(y, y_pred_oof, pos_label=0, zero_division=0), 6)
summary["oof_f1_macro"] = round(f1_score(y, y_pred_oof, average="macro", zero_division=0), 6)
summary["oof_roc_auc"] = round(roc_auc_score(y, y_proba_oof), 6)

SUMMARY_FILE = BASE_DIR / "rfm_experiment_1_summary.csv"
pd.DataFrame([summary], index=["rfm_experiment_1"]).to_csv(SUMMARY_FILE, index=True)
print(f"\nSummary saved: {SUMMARY_FILE}")

print("\n" + "=" * 70)
print("Experiment complete.")
print("=" * 70)
