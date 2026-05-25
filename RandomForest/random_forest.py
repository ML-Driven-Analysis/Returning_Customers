import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier  # Model RandomForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
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
# 2. Basic dataset validation
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
# 4. Feature extraction
# =========================================================

def extract_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts useful model features from the original e-commerce dataset.
    The function keeps customer behavior data and removes raw identifiers later.
    """

    data = data.copy()

    # -------------------------
    # Date features
    # -------------------------
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["DayOfWeek"] = data["Date"].dt.dayofweek
    data["DayOfMonth"] = data["Date"].dt.day
    data["Quarter"] = data["Date"].dt.quarter

    # Saturday = 5, Sunday = 6 according to pandas dayofweek
    data["IsWeekend"] = data["DayOfWeek"].isin([5, 6]).astype(int)

    # -------------------------
    # Purchase amount features
    # -------------------------
    data["Original_Amount"] = data["Unit_Price"] * data["Quantity"]

    data["Has_Discount"] = (data["Discount_Amount"] > 0).astype(int)

    data["Discount_Rate"] = np.where(
        data["Original_Amount"] > 0,
        data["Discount_Amount"] / data["Original_Amount"],
        0
    )

    data["Amount_Per_Item"] = np.where(
        data["Quantity"] > 0,
        data["Total_Amount"] / data["Quantity"],
        0
    )

    # -------------------------
    # Website behavior features
    # -------------------------
    data["Pages_Per_Minute"] = np.where(
        data["Session_Duration_Minutes"] > 0,
        data["Pages_Viewed"] / data["Session_Duration_Minutes"],
        0
    )

    data["Long_Session"] = (
        data["Session_Duration_Minutes"] > data["Session_Duration_Minutes"].median()
    ).astype(int)

    data["Many_Pages_Viewed"] = (
        data["Pages_Viewed"] > data["Pages_Viewed"].median()
    ).astype(int)

    # -------------------------
    # Delivery and rating features
    # -------------------------
    data["Fast_Delivery"] = (
        data["Delivery_Time_Days"] <= data["Delivery_Time_Days"].median()
    ).astype(int)

    data["High_Rating"] = (data["Customer_Rating"] >= 4).astype(int)

    # Drop raw Date after extracting useful date features
    data = data.drop(columns=["Date"])

    return data


df = extract_features(df)


# =========================================================
# 5. Remove columns that should not enter the model
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
    "Unnamed: 0",

    #Run 2 - drop Price related features
    "Unit_Price",
    "Original_Amount",
    "Amount_Per_Item",
    "Total_Amount",
    "Discount_Amount", 
    "Has_Discount", 
    "Discount_Rate"

]

columns_to_drop = [col for col in columns_to_drop if col in df.columns]

X = df.drop(columns=columns_to_drop)
y = df[label_col]

print("\nFeatures used by the model:")
print(X.columns.tolist())

print("\nNumber of features before One-Hot Encoding:", X.shape[1])


# =========================================================
# 6. Detect numeric and categorical columns
# =========================================================

numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# =========================================================
# 7. Preprocessing
# =========================================================

# Decision trees do not require scaling, so StandardScaler is not necessary.
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
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


# =========================================================
# 8. Random Forest model
# =========================================================

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,      # Number of trees in the forest
        max_depth=10,          # Depth limit to prevent overfitting
        class_weight="balanced", #Run 3 - returning: 0.835, new: 1.243
        random_state=42,       # Ensures reproducible results across runs
        n_jobs=-1              # Uses all available CPU cores for faster execution
    ))
])


# =========================================================
# 9. Train / Test split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)


# =========================================================
# 10. Train model
# =========================================================

model.fit(X_train, y_train)


# =========================================================
# 11. Predict (Run 1-3)
# =========================================================

# y_pred = model.predict(X_test)
# y_proba = model.predict_proba(X_test)[:, 1]

# =========================================================
# 11. Predict with Custom Threshold (58%) - Run 4
# =========================================================

# A. Get the predicted probabilities (confidence levels) for each customer belonging to Class 1 (Returning Customer)
y_proba = model.predict_proba(X_test)[:, 1]

# B. Set the new, stricter custom probability threshold (58% instead of the default 50%)
custom_threshold = 0.58

# C. Classify as 1 (Returning Customer) only if the predicted probability meets or exceeds the custom threshold
y_pred = (y_proba >= custom_threshold).astype(int)

# =========================================================
# 12. Evaluation
# =========================================================

# Run 1-3
# print("\n==============================")
# print("Model Evaluation - Random Forest")
# print("==============================")

print("\n==============================")
print(f"Model Evaluation - Random Forest (Threshold: {custom_threshold * 100}%)") # Run 4
print("==============================")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# ========================================================================
# 13. Show Feature Importances - according to Importances at all the trees
# ========================================================================

try:
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    print("\nTop 15 most important features in Random Forest:")
    print(
        importance_df
        .sort_values(by="Importance", ascending=False)
        .head(15)
        .to_string(index=False)
    )

except Exception as e:
    print("\nCould not print feature importances:")
    print(e)


# =========================================================
# 14. Save trained model
# =========================================================

MODEL_FILE = BASE_DIR / "random_forest_returning_customer_model.pkl"

joblib.dump(model, MODEL_FILE)

print("\nModel saved as:")
print(MODEL_FILE)