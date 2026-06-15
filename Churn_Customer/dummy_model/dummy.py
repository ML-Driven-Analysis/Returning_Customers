"""
Churn Customer — Dummy Baseline (Lower Bound)

מודל דמי שמשמש כחסם תחתון להשוואה עם כל המודלים האמיתיים.
מריץ שלוש אסטרטגיות:
  - most_frequent : תמיד מנבא 0 (נשאר) — הכי נפוץ בדאטה
  - stratified    : מנבא לפי התפלגות המחלקות בדאטה
  - uniform       : מנבא 0 או 1 בהסתברות שווה

Data source : E Commerce Dataset.xlsx  (Sheet: E Comm)
Target      : Churn  (1 = churned, 0 = retained)
Validation  : StratifiedKFold (5 splits, shuffle=True, random_state=42)
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

# Paths

BASE_DIR  = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "dataset" / "E Commerce Dataset.xlsx"

_UPLOAD_PATH = Path("/mnt/user-data/uploads/E_Commerce_Dataset.xlsx")
if not DATA_FILE.exists() and _UPLOAD_PATH.exists():
    DATA_FILE = _UPLOAD_PATH

print("=" * 70)
print("Churn Customer — Dummy Baseline (Lower Bound)")
print("=" * 70)
print(f"\nDataset: {DATA_FILE}")

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")


# Load & validate

df = pd.read_excel(DATA_FILE, sheet_name="E Comm")
df.columns = df.columns.str.strip()

LABEL = "Churn"
df[LABEL] = pd.to_numeric(df[LABEL], errors="coerce")
df = df.dropna(subset=[LABEL])
df[LABEL] = df[LABEL].astype(int)

n0 = (df[LABEL] == 0).sum()
n1 = (df[LABEL] == 1).sum()
print(f"\nDataset: {df.shape[0]:,} rows")
print(f"  Churn = 0 (retained): {n0:,} ({n0/len(df)*100:.1f}%)")
print(f"  Churn = 1 (churned):  {n1:,} ({n1/len(df)*100:.1f}%)")

# DummyClassifier לא צריך פיצ'רים, אבל sklearn מצפה ל-X
X = df.drop(columns=[LABEL])
y = df[LABEL]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# הרצה והדפסה לכל אסטרטגיה

STRATEGIES = ["most_frequent", "stratified", "uniform"]
rows = []

for strategy in STRATEGIES:
    dummy = DummyClassifier(strategy=strategy, random_state=42)

    y_pred  = cross_val_predict(dummy, X, y, cv=cv, method="predict",       n_jobs=1)
    y_proba = cross_val_predict(dummy, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]

    metrics = {
        "strategy":          strategy,
        "accuracy":          round(accuracy_score(y, y_pred),                              4),
        "balanced_accuracy": round(balanced_accuracy_score(y, y_pred),                     4),
        "precision_1":       round(precision_score(y, y_pred, pos_label=1, zero_division=0), 4),
        "recall_1":          round(recall_score(y, y_pred,    pos_label=1, zero_division=0), 4),
        "f1_1":              round(f1_score(y, y_pred,        pos_label=1, zero_division=0), 4),
        "f1_macro":          round(f1_score(y, y_pred, average="macro",    zero_division=0), 4),
        "roc_auc":           round(roc_auc_score(y, y_proba),              4),
        "pr_auc":            round(average_precision_score(y, y_proba),    4),
    }
    rows.append(metrics)

    cm = confusion_matrix(y, y_pred)

    print(f"\n{'=' * 70}")
    print(f"Strategy: {strategy}")
    print(f"{'=' * 70}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Precision (Ch=1):  {metrics['precision_1']:.4f}")
    print(f"  Recall    (Ch=1):  {metrics['recall_1']:.4f}")
    print(f"  F1        (Ch=1):  {metrics['f1_1']:.4f}")
    print(f"  F1 Macro:          {metrics['f1_macro']:.4f}")
    print(f"  ROC AUC:           {metrics['roc_auc']:.4f}")
    print(f"  PR AUC:            {metrics['pr_auc']:.4f}")
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"               Pred 0   Pred 1")
    print(f"  Actual 0  :  {cm[0][0]:>6}   {cm[0][1]:>6}")
    print(f"  Actual 1  :  {cm[1][0]:>6}   {cm[1][1]:>6}")



# השוואה מרוכזת

print(f"\n{'=' * 70}")
print("השוואה מרוכזת — כל האסטרטגיות")
print(f"{'=' * 70}")
print(f"  {'מדד':<22} {'most_frequent':>15} {'stratified':>12} {'uniform':>10}")
print(f"  {'-' * 62}")

metrics_order = [
    "accuracy", "balanced_accuracy", "precision_1",
    "recall_1", "f1_1", "f1_macro", "roc_auc", "pr_auc"
]
for m in metrics_order:
    vals = [str(r[m]) for r in rows]
    print(f"  {m:<22} {vals[0]:>15} {vals[1]:>12} {vals[2]:>10}")


# Save summary

SUMMARY_FILE = BASE_DIR / "dummy_baseline_summary.csv"
pd.DataFrame(rows).set_index("strategy").to_csv(SUMMARY_FILE)
print(f"\nSummary saved → {SUMMARY_FILE}")

print(f"\n{'=' * 70}")
print("Dummy Baseline complete.")
print(f"{'=' * 70}")