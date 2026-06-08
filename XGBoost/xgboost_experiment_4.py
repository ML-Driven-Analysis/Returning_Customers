"""
Experiment 4: Feature engineering optimization + threshold tuning.

Two sub-experiments:
  4a — Remove redundant binary features, evaluate if feature reduction helps.
  4b — Use default features with optimal threshold from Precision-Recall curve.

Best configuration from each sub-experiment is compared against baseline.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
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
    precision_recall_curve
)
from xgboost import XGBClassifier


# =========================================================
# 1. Load dataset
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "dataset" / "ecommerce_customer_behavior_dataset.csv"

print("Running script from:")
print(BASE_DIR)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Could not find dataset file: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)


# =========================================================
# 2. Dataset validation + label conversion
# =========================================================

label_col = "Is_Returning_Customer"

required_columns = [
    "Order_ID", "Customer_ID", "Date", "Age", "Gender", "City",
    "Product_Category", "Unit_Price", "Quantity", "Discount_Amount",
    "Total_Amount", "Payment_Method", "Device_Type",
    "Session_Duration_Minutes", "Pages_Viewed", "Is_Returning_Customer",
    "Delivery_Time_Days", "Customer_Rating"
]

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"The following required columns are missing: {missing_columns}")


def convert_label(value):
    value_str = str(value).strip().lower()
    if value_str in ["true", "1", "yes", "returning", "returning customer"]:
        return 1
    if value_str in ["false", "0", "no", "not returning", "not returning customer"]:
        return 0
    raise ValueError(f"Unknown label value: {value}")


df[label_col] = df[label_col].apply(convert_label)

n_class_0 = (df[label_col] == 0).sum()
n_class_1 = (df[label_col] == 1).sum()
scale_pos_weight = n_class_0 / n_class_1
print(f"\nLabel distribution: class 0={n_class_0}, class 1={n_class_1}")
print(f"scale_pos_weight = {scale_pos_weight:.4f}")


# =========================================================
# 3. Separate X and y
# =========================================================

columns_to_drop = [
    label_col, "Order_ID", "Customer_ID", "Serial_Number", "serial_number",
    "Serial", "serial", "Index", "index", "Unnamed: 0"
]
columns_to_drop = [col for col in columns_to_drop if col in df.columns]

X_raw = df.drop(columns=columns_to_drop)
y = df[label_col]


# =========================================================
# 4. Custom feature extractor
# =========================================================

class EcommerceFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, remove_redundant_features=False):
        self.remove_redundant_features = remove_redundant_features

    def fit(self, X, y=None):
        data = X.copy()
        for col in ["Session_Duration_Minutes", "Pages_Viewed", "Delivery_Time_Days"]:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")

        self.session_duration_median_ = data["Session_Duration_Minutes"].median() if "Session_Duration_Minutes" in data.columns else 0
        self.pages_viewed_median_ = data["Pages_Viewed"].median() if "Pages_Viewed" in data.columns else 0
        self.delivery_time_median_ = data["Delivery_Time_Days"].median() if "Delivery_Time_Days" in data.columns else 0
        return self

    def transform(self, X):
        data = X.copy()

        for col in ["Age", "Unit_Price", "Quantity", "Discount_Amount", "Total_Amount",
                    "Session_Duration_Minutes", "Pages_Viewed", "Delivery_Time_Days", "Customer_Rating"]:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")

        if "Date" in data.columns:
            data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
            data["Year"] = data["Date"].dt.year
            data["Month"] = data["Date"].dt.month
            data["DayOfWeek"] = data["Date"].dt.dayofweek
            data["DayOfMonth"] = data["Date"].dt.day
            data["Quarter"] = data["Date"].dt.quarter
            data["IsWeekend"] = data["DayOfWeek"].isin([5, 6]).astype(int)
            data = data.drop(columns=["Date"])

        if "Unit_Price" in data.columns and "Quantity" in data.columns:
            data["Original_Amount"] = data["Unit_Price"] * data["Quantity"]

        if "Discount_Amount" in data.columns:
            data["Has_Discount"] = (data["Discount_Amount"] > 0).astype(int)

        if "Original_Amount" in data.columns and "Discount_Amount" in data.columns:
            data["Discount_Rate"] = np.where(
                data["Original_Amount"] > 0,
                data["Discount_Amount"] / data["Original_Amount"], 0
            )

        if "Total_Amount" in data.columns and "Quantity" in data.columns:
            data["Amount_Per_Item"] = np.where(
                data["Quantity"] > 0, data["Total_Amount"] / data["Quantity"], 0
            )

        if "Pages_Viewed" in data.columns and "Session_Duration_Minutes" in data.columns:
            data["Pages_Per_Minute"] = np.where(
                data["Session_Duration_Minutes"] > 0,
                data["Pages_Viewed"] / data["Session_Duration_Minutes"], 0
            )
            data["Long_Session"] = (data["Session_Duration_Minutes"] > self.session_duration_median_).astype(int)
            data["Many_Pages_Viewed"] = (data["Pages_Viewed"] > self.pages_viewed_median_).astype(int)

        if "Delivery_Time_Days" in data.columns:
            data["Fast_Delivery"] = (data["Delivery_Time_Days"] <= self.delivery_time_median_).astype(int)

        if "Customer_Rating" in data.columns:
            data["High_Rating"] = (data["Customer_Rating"] >= 4).astype(int)

        if self.remove_redundant_features:
            redundant = ["Original_Amount", "Long_Session", "Many_Pages_Viewed", "Fast_Delivery", "High_Rating"]
            data = data.drop(columns=[c for c in redundant if c in data.columns])

        return data


# =========================================================
# 5. Pipeline builder
# =========================================================

def get_feature_types(remove_redundant_features=False):
    extractor = EcommerceFeatureExtractor(remove_redundant_features=remove_redundant_features)
    sample = extractor.fit_transform(X_raw)
    numeric_features = sample.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = sample.select_dtypes(include=["object", "str", "category", "bool"]).columns.tolist()
    return numeric_features, categorical_features


def build_pipeline(remove_redundant_features=False, spw=scale_pos_weight):
    numeric_features, categorical_features = get_feature_types(remove_redundant_features=remove_redundant_features)

    numeric_transformer = SimpleImputer(strategy="median")

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    model = Pipeline(steps=[
        ("feature_extractor", EcommerceFeatureExtractor(remove_redundant_features=remove_redundant_features)),
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            random_state=42,
            eval_metric="logloss",
            verbosity=0
        ))
    ])
    return model


# =========================================================
# 6. CV setup
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
    "roc_auc": "roc_auc"
}


def run_cv_and_print(model, label):
    print(f"\n{'=' * 70}")
    print(f"Sub-experiment: {label}")
    print("=" * 70)

    cv_results = cross_validate(
        model, X_raw, y,
        cv=cv, scoring=scoring, n_jobs=1, return_train_score=False
    )

    print("\nCross Validation results (5-Fold):")
    for metric_name in scoring.keys():
        values = cv_results[f"test_{metric_name}"]
        print(f"{metric_name}: {values.mean():.4f} ± {values.std():.4f}")

    y_pred_cv = cross_val_predict(model, X_raw, y, cv=cv, method="predict", n_jobs=1)
    y_proba_cv = cross_val_predict(model, X_raw, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]

    print("\nAggregated out-of-fold metrics:")
    print("Accuracy:", accuracy_score(y, y_pred_cv))
    print("Precision class 0:", precision_score(y, y_pred_cv, pos_label=0, zero_division=0))
    print("Recall class 0:", recall_score(y, y_pred_cv, pos_label=0, zero_division=0))
    print("F1 class 0:", f1_score(y, y_pred_cv, pos_label=0, zero_division=0))
    print("Precision class 1:", precision_score(y, y_pred_cv, pos_label=1, zero_division=0))
    print("Recall class 1:", recall_score(y, y_pred_cv, pos_label=1, zero_division=0))
    print("F1 class 1:", f1_score(y, y_pred_cv, pos_label=1, zero_division=0))
    print("F1 Macro:", f1_score(y, y_pred_cv, average="macro", zero_division=0))
    print("ROC AUC:", roc_auc_score(y, y_proba_cv))

    print("\nAggregated Confusion Matrix:")
    print(confusion_matrix(y, y_pred_cv))

    print("\nClassification Report:")
    print(classification_report(y, y_pred_cv, zero_division=0))

    return cv_results, y_pred_cv, y_proba_cv


# =========================================================
# 7. Sub-experiment 4a — Remove redundant features
# =========================================================

model_4a = build_pipeline(remove_redundant_features=True, spw=scale_pos_weight)
cv_results_4a, y_pred_4a, y_proba_4a = run_cv_and_print(model_4a, "4a — Reduced Features + scale_pos_weight")


# =========================================================
# 8. Sub-experiment 4b — Threshold tuning (full features)
# =========================================================

print(f"\n{'=' * 70}")
print("Sub-experiment 4b — Threshold Tuning on Precision-Recall Curve")
print("=" * 70)

model_4b = build_pipeline(remove_redundant_features=False, spw=scale_pos_weight)

y_proba_4b = cross_val_predict(model_4b, X_raw, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]

# Search for threshold that maximizes F1 Macro
precisions, recalls, thresholds = precision_recall_curve(y, y_proba_4b)

best_threshold = 0.5
best_f1_macro = 0.0

for t in thresholds:
    y_pred_t = (y_proba_4b >= t).astype(int)
    f1_m = f1_score(y, y_pred_t, average="macro", zero_division=0)
    if f1_m > best_f1_macro:
        best_f1_macro = f1_m
        best_threshold = t

print(f"\nOptimal threshold: {best_threshold:.4f}")
print(f"Best F1 Macro at optimal threshold: {best_f1_macro:.4f}")

y_pred_optimal = (y_proba_4b >= best_threshold).astype(int)

print("\nMetrics at optimal threshold:")
print("Accuracy:", accuracy_score(y, y_pred_optimal))
print("Precision class 0:", precision_score(y, y_pred_optimal, pos_label=0, zero_division=0))
print("Recall class 0:", recall_score(y, y_pred_optimal, pos_label=0, zero_division=0))
print("F1 class 0:", f1_score(y, y_pred_optimal, pos_label=0, zero_division=0))
print("Precision class 1:", precision_score(y, y_pred_optimal, pos_label=1, zero_division=0))
print("Recall class 1:", recall_score(y, y_pred_optimal, pos_label=1, zero_division=0))
print("F1 class 1:", f1_score(y, y_pred_optimal, pos_label=1, zero_division=0))
print("F1 Macro:", f1_score(y, y_pred_optimal, average="macro", zero_division=0))
print("ROC AUC:", roc_auc_score(y, y_proba_4b))

print("\nConfusion Matrix (optimal threshold):")
print(confusion_matrix(y, y_pred_optimal))

print("\nClassification Report (optimal threshold):")
print(classification_report(y, y_pred_optimal, zero_division=0))

# Save threshold analysis
threshold_results = []
for t in np.arange(0.1, 0.9, 0.05):
    y_pred_t = (y_proba_4b >= t).astype(int)
    threshold_results.append({
        "threshold": round(t, 2),
        "accuracy": accuracy_score(y, y_pred_t),
        "recall_0": recall_score(y, y_pred_t, pos_label=0, zero_division=0),
        "recall_1": recall_score(y, y_pred_t, pos_label=1, zero_division=0),
        "f1_0": f1_score(y, y_pred_t, pos_label=0, zero_division=0),
        "f1_1": f1_score(y, y_pred_t, pos_label=1, zero_division=0),
        "f1_macro": f1_score(y, y_pred_t, average="macro", zero_division=0),
    })

threshold_df = pd.DataFrame(threshold_results)
THRESHOLD_FILE = BASE_DIR / "xgboost_experiment_4_threshold_analysis.csv"
threshold_df.to_csv(THRESHOLD_FILE, index=False)
print(f"\nThreshold analysis saved as:\n{THRESHOLD_FILE}")

print("\nThreshold vs F1 Macro:")
print(threshold_df[["threshold", "recall_0", "recall_1", "f1_macro"]].to_string(index=False))


# =========================================================
# 9. Save CV summary
# =========================================================

SUMMARY_FILE = BASE_DIR / "xgboost_cv_summary.csv"

cv_4b_results = cross_validate(
    model_4b, X_raw, y,
    cv=cv, scoring=scoring, n_jobs=1, return_train_score=False
)

all_summaries = {}
for exp_label, cv_res in [("xgboost_experiment_4a", cv_results_4a), ("xgboost_experiment_4b", cv_4b_results)]:
    summary = {}
    for metric_name in scoring.keys():
        values = cv_res[f"test_{metric_name}"]
        summary[f"{metric_name}_mean"] = values.mean()
        summary[f"{metric_name}_std"] = values.std()
    all_summaries[exp_label] = summary

if SUMMARY_FILE.exists():
    existing = pd.read_csv(SUMMARY_FILE, index_col=0)
    new_rows = pd.DataFrame(all_summaries).T
    combined = pd.concat([existing, new_rows])
    combined.to_csv(SUMMARY_FILE, index=True)
else:
    pd.DataFrame(all_summaries).T.to_csv(SUMMARY_FILE, index=True)

print("\nCV summary appended to:")
print(SUMMARY_FILE)

print("\n" + "=" * 70)
print("Experiment 4 complete.")
print("=" * 70)
