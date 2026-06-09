import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
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
    make_scorer
)


# =========================================================
# 1. Load dataset
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR.parent / "dataset" / "ecommerce_customer_behavior_dataset.csv"

print("Running script from:")
print(BASE_DIR)

print("\nTrying to load dataset from:")
print(DATA_FILE)

if not DATA_FILE.exists():
    dataset_dir = BASE_DIR.parent / "dataset"

    print("\nDataset file was not found.")
    print("Expected dataset folder:")
    print(dataset_dir)

    if dataset_dir.exists():
        print("\nFiles inside dataset folder:")
        for file in dataset_dir.iterdir():
            print(file.name)
    else:
        print("\nDataset folder does not exist.")

    raise FileNotFoundError(f"Could not find dataset file: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)
print("\nDataset columns:")
print(df.columns.tolist())


# =========================================================
# 2. Dataset validation
# =========================================================

label_col = "Is_Returning_Customer"

required_columns = [
    "Order_ID",
    "Customer_ID",
    "Date",
    "Age",
    "Gender",
    "City",
    "Product_Category",
    "Unit_Price",
    "Quantity",
    "Discount_Amount",
    "Total_Amount",
    "Payment_Method",
    "Device_Type",
    "Session_Duration_Minutes",
    "Pages_Viewed",
    "Is_Returning_Customer",
    "Delivery_Time_Days",
    "Customer_Rating"
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"The following required columns are missing: {missing_columns}")

print("\nMissing values per column:")
print(df.isna().sum())


# =========================================================
# 3. Convert label to 0 / 1
# =========================================================

def convert_label(value):
    value_str = str(value).strip().lower()

    if value_str in ["true", "1", "yes", "returning", "returning customer"]:
        return 1

    if value_str in ["false", "0", "no", "not returning", "not returning customer"]:
        return 0

    raise ValueError(f"Unknown label value: {value}")


df[label_col] = df[label_col].apply(convert_label)

print("\nLabel distribution:")
print(df[label_col].value_counts())

print("\nLabel distribution percentage:")
print(df[label_col].value_counts(normalize=True) * 100)


# =========================================================
# 4. Separate X and y
# =========================================================

columns_to_drop = [
    label_col,
    "Order_ID",
    "Customer_ID",
    "Serial_Number",
    "serial_number",
    "Serial",
    "serial",
    "Index",
    "index",
    "Unnamed: 0"
]

columns_to_drop = [col for col in columns_to_drop if col in df.columns]

X_raw = df.drop(columns=columns_to_drop)
y = df[label_col]


# =========================================================
# 5. Custom feature extractor
# =========================================================

class EcommerceFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Feature engineering transformer.

    Important:
    Median-based features are calculated inside fit().
    This prevents data leakage during Cross Validation.
    """

    def __init__(self, remove_redundant_features=False):
        self.remove_redundant_features = remove_redundant_features

    def fit(self, X, y=None):
        data = X.copy()

        numeric_cols = [
            "Session_Duration_Minutes",
            "Pages_Viewed",
            "Delivery_Time_Days"
        ]

        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")

        self.session_duration_median_ = (
            data["Session_Duration_Minutes"].median()
            if "Session_Duration_Minutes" in data.columns
            else 0
        )

        self.pages_viewed_median_ = (
            data["Pages_Viewed"].median()
            if "Pages_Viewed" in data.columns
            else 0
        )

        self.delivery_time_median_ = (
            data["Delivery_Time_Days"].median()
            if "Delivery_Time_Days" in data.columns
            else 0
        )

        return self

    def transform(self, X):
        data = X.copy()

        # -------------------------
        # Convert numeric columns safely
        # -------------------------
        numeric_cols = [
            "Age",
            "Unit_Price",
            "Quantity",
            "Discount_Amount",
            "Total_Amount",
            "Session_Duration_Minutes",
            "Pages_Viewed",
            "Delivery_Time_Days",
            "Customer_Rating"
        ]

        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")

        # -------------------------
        # Date features
        # -------------------------
        if "Date" in data.columns:
            data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

            data["Year"] = data["Date"].dt.year
            data["Month"] = data["Date"].dt.month
            data["DayOfWeek"] = data["Date"].dt.dayofweek
            data["DayOfMonth"] = data["Date"].dt.day
            data["Quarter"] = data["Date"].dt.quarter
            data["IsWeekend"] = data["DayOfWeek"].isin([5, 6]).astype(int)

            data = data.drop(columns=["Date"])

        # -------------------------
        # Purchase amount features
        # -------------------------
        if "Unit_Price" in data.columns and "Quantity" in data.columns:
            data["Original_Amount"] = data["Unit_Price"] * data["Quantity"]

        if "Discount_Amount" in data.columns:
            data["Has_Discount"] = (data["Discount_Amount"] > 0).astype(int)

        if "Original_Amount" in data.columns and "Discount_Amount" in data.columns:
            data["Discount_Rate"] = np.where(
                data["Original_Amount"] > 0,
                data["Discount_Amount"] / data["Original_Amount"],
                0
            )

        if "Total_Amount" in data.columns and "Quantity" in data.columns:
            data["Amount_Per_Item"] = np.where(
                data["Quantity"] > 0,
                data["Total_Amount"] / data["Quantity"],
                0
            )

        # -------------------------
        # Website behavior features
        # -------------------------
        if "Pages_Viewed" in data.columns and "Session_Duration_Minutes" in data.columns:
            data["Pages_Per_Minute"] = np.where(
                data["Session_Duration_Minutes"] > 0,
                data["Pages_Viewed"] / data["Session_Duration_Minutes"],
                0
            )

            data["Long_Session"] = (
                data["Session_Duration_Minutes"] > self.session_duration_median_
            ).astype(int)

            data["Many_Pages_Viewed"] = (
                data["Pages_Viewed"] > self.pages_viewed_median_
            ).astype(int)

        # -------------------------
        # Delivery and rating features
        # -------------------------
        if "Delivery_Time_Days" in data.columns:
            data["Fast_Delivery"] = (
                data["Delivery_Time_Days"] <= self.delivery_time_median_
            ).astype(int)

        if "Customer_Rating" in data.columns:
            data["High_Rating"] = (data["Customer_Rating"] >= 4).astype(int)

        # -------------------------
        # Experiment 2 feature removal
        # -------------------------
        if self.remove_redundant_features:
            redundant_features = [
                "Original_Amount",
                "Long_Session",
                "Many_Pages_Viewed",
                "Fast_Delivery",
                "High_Rating"
            ]

            redundant_features = [
                col for col in redundant_features if col in data.columns
            ]

            data = data.drop(columns=redundant_features)

        return data


# =========================================================
# 6. Build model pipeline
# =========================================================

def get_feature_types(remove_redundant_features):
    """
    Creates a temporary transformed sample only to detect
    numeric and categorical feature names.
    """

    extractor = EcommerceFeatureExtractor(
        remove_redundant_features=remove_redundant_features
    )

    sample = extractor.fit_transform(X_raw)

    numeric_features = sample.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = sample.select_dtypes(
        include=["object", "str", "category", "bool"]
    ).columns.tolist()

    return numeric_features, categorical_features


def build_logistic_pipeline(remove_redundant_features=False, class_weight=None):
    numeric_features, categorical_features = get_feature_types(
        remove_redundant_features=remove_redundant_features
    )

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    model = Pipeline(steps=[
        ("feature_extractor", EcommerceFeatureExtractor(
            remove_redundant_features=remove_redundant_features
        )),
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=3000,
            random_state=42,
            class_weight=class_weight
        ))
    ])

    return model


# =========================================================
# 7. Cross Validation setup
# =========================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

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


# =========================================================
# 8. Experiments
# =========================================================

experiments = {
    "baseline_all_features": {
        "remove_redundant_features": False,
        "class_weight": None
    },
    "balanced_all_features": {
        "remove_redundant_features": False,
        "class_weight": "balanced"
    },
    "balanced_reduced_features": {
        "remove_redundant_features": True,
        "class_weight": "balanced"
    }
}


summary_rows = {}
trained_model_templates = {}

for experiment_name, params in experiments.items():
    print("\n" + "=" * 70)
    print(f"Running experiment: {experiment_name}")
    print("=" * 70)

    model = build_logistic_pipeline(
        remove_redundant_features=params["remove_redundant_features"],
        class_weight=params["class_weight"]
    )

    trained_model_templates[experiment_name] = model

    cv_results = cross_validate(
        model,
        X_raw,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False
    )

    row = {}

    for metric_name in scoring.keys():
        values = cv_results[f"test_{metric_name}"]
        row[f"{metric_name}_mean"] = values.mean()
        row[f"{metric_name}_std"] = values.std()

    summary_rows[experiment_name] = row

    print("\nCross Validation results:")
    for metric_name in scoring.keys():
        mean_value = row[f"{metric_name}_mean"]
        std_value = row[f"{metric_name}_std"]
        print(f"{metric_name}: {mean_value:.4f} ± {std_value:.4f}")

    # Out-of-fold predictions for aggregated confusion matrix
    y_pred_cv = cross_val_predict(
        model,
        X_raw,
        y,
        cv=cv,
        method="predict",
        n_jobs=1
    )

    y_proba_cv = cross_val_predict(
        model,
        X_raw,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=1
    )[:, 1]

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


# =========================================================
# 9. Summary table
# =========================================================

summary_df = pd.DataFrame(summary_rows).T

important_columns = [
    "accuracy_mean",
    "balanced_accuracy_mean",
    "recall_0_mean",
    "recall_1_mean",
    "f1_0_mean",
    "f1_1_mean",
    "f1_macro_mean",
    "roc_auc_mean"
]

print("\n" + "=" * 70)
print("Cross Validation Summary")
print("=" * 70)

print(summary_df[important_columns].sort_values(
    by=["f1_macro_mean", "roc_auc_mean"],
    ascending=False
).to_string())


SUMMARY_FILE = BASE_DIR / "logistic_cross_validation_summary.csv"
summary_df.to_csv(SUMMARY_FILE, index=True)

print("\nCross Validation summary saved as:")
print(SUMMARY_FILE)


# =========================================================
# 10. Choose best model by F1 Macro
# =========================================================

best_experiment_name = summary_df["f1_macro_mean"].idxmax()

print("\nBest experiment by F1 Macro:")
print(best_experiment_name)

best_params = experiments[best_experiment_name]

best_model = build_logistic_pipeline(
    remove_redundant_features=best_params["remove_redundant_features"],
    class_weight=best_params["class_weight"]
)

best_model.fit(X_raw, y)


# =========================================================
# 11. Coefficients of best model
# =========================================================

try:
    feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = best_model.named_steps["classifier"].coef_[0]

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients
    })

    coef_df["Abs_Coefficient"] = coef_df["Coefficient"].abs()

    print("\nTop 15 most influential features in best model:")
    print(
        coef_df
        .sort_values(by="Abs_Coefficient", ascending=False)
        .head(15)
        [["Feature", "Coefficient"]]
        .to_string(index=False)
    )

    COEF_FILE = BASE_DIR / "best_logistic_coefficients.csv"
    coef_df.sort_values(by="Abs_Coefficient", ascending=False).to_csv(
        COEF_FILE,
        index=False
    )

    print("\nBest model coefficients saved as:")
    print(COEF_FILE)

except Exception as e:
    print("\nCould not print coefficients:")
    print(e)