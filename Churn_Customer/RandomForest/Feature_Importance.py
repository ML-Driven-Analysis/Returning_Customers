"""
Churn Customer — Random Forest Feature Importance Extraction

This script trains the final pipeline on the full dataset to extract
and save the global feature importances.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# =========================================================
# 1. Load dataset
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "dataset" / "E Commerce Dataset.xlsx"

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_excel(DATA_FILE, sheet_name="E Comm")
df.columns = df.columns.str.strip()

# Drop rows with missing values in the target column and convert to integer
df = df.dropna(subset=["Churn"])
df["Churn"] = df["Churn"].astype(int)

# =========================================================
# 2. Feature selection & column grouping
# =========================================================
X = df.drop(columns=["CustomerID", "Churn"], errors="ignore")
y = df["Churn"]

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

# Ensure categorical columns are treated as clean strings to prevent imputer issues
for col in categorical_features:
    X[col] = X[col].astype(str)

# =========================================================
# 3. Build data preprocessing pipeline & train model
# =========================================================
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
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
        n_jobs=-1                # Uses all available CPU cores
    )),
])

print("Training final model on full dataset for feature importance extraction...")
model.fit(X, y)

# =========================================================
# 4. Extract transformed feature names and importances
# =========================================================
# Get categorical feature names after One-Hot Encoding transformation
cat_encoder = model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
encoded_cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()

# Combine numeric and encoded categorical names in the exact alignment order
all_feature_names = numeric_features + encoded_cat_names

# Extract feature importance weights from the classifier
importances = model.named_steps["classifier"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": all_feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# =========================================================
# 5. Print & Save Results
# =========================================================
print("\n" + "=" * 60)
print("GLOBAL FEATURE IMPORTANCE RANKING (Random Forest)")
print("=" * 60)
print(importance_df.to_string(index=False))

# Save the feature ranking matrix to a summary CSV file
IMPORTANCE_FILE = BASE_DIR / "random_forest_feature_importances.csv"
importance_df.to_csv(IMPORTANCE_FILE, index=False)
print(f"\nFeature importances saved successfully to: {IMPORTANCE_FILE}")