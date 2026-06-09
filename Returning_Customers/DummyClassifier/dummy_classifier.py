import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
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
    raise FileNotFoundError(f"Could not find dataset file: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)


# =========================================================
# 2. Convert label to 0 / 1
# =========================================================

label_col = "Is_Returning_Customer"


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
# 3. Features and label
# DummyClassifier ignores X but requires it for the sklearn API
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

X = df.drop(columns=columns_to_drop)
y = df[label_col]


# =========================================================
# 4. Train / Test split — identical to logistic baseline
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
# 5. DummyClassifier model
# =========================================================

model = DummyClassifier(strategy="most_frequent", random_state=42)
model.fit(X_train, y_train)


# =========================================================
# 6. Predict
# =========================================================

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]


# =========================================================
# 7. Evaluation
# =========================================================

print("\n==============================")
print("Model Evaluation")
print("==============================")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))

try:
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
except ValueError as e:
    print("ROC AUC: N/A (constant predictions —", e, ")")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# =========================================================
# 8. Save model
# =========================================================

MODEL_FILE = BASE_DIR / "dummy_classifier_model.pkl"

joblib.dump(model, MODEL_FILE)

print("\nModel saved as:")
print(MODEL_FILE)
