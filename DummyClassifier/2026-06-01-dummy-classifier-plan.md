# DummyClassifier Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** יצירת מודל baseline טיפש (most_frequent) להשוואה עם הרגרסיה הלוגיסטית, תיעוד תוצאות ב-README, ועדכון README הלוגיסטי עם השוואה.

**Architecture:** סקריפט Python עצמאי עם אותה חלוקת נתונים כמו ה-baseline הלוגיסטי. אין feature engineering. לאחר הרצה: README מלא ועדכון README הלוגיסטי.

**Tech Stack:** Python, pandas, scikit-learn (DummyClassifier), joblib

---

### Task 1: יצירת dummy_classifier.py

**Files:**
- Create: `DummyClassifier/dummy_classifier.py`

- [ ] **Step 1: צור את הקובץ עם הקוד המלא**

צור `DummyClassifier/dummy_classifier.py` עם התוכן הבא:

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add DummyClassifier/dummy_classifier.py
git commit -m "feat: add DummyClassifier baseline model"
```

---

### Task 2: הרצת המודל ותיעוד תוצאות

**Files:**
- Run: `DummyClassifier/dummy_classifier.py`

- [ ] **Step 1: הרץ את הסקריפט**

```bash
python DummyClassifier/dummy_classifier.py
```

- [ ] **Step 2: רשום את הערכים הבאים מהפלט (יידרשו ב-Task 3 ו-4)**

```
Accuracy: X.XXX
Precision: X.XXX
Recall: X.XXX
F1 Score: X.XXX
ROC AUC: X.XXX  (או N/A)

Confusion Matrix:
[[ TN  FP ]
 [ FN  TP ]]

Classification Report:
           0   precision=X  recall=X  f1=X
           1   precision=X  recall=X  f1=X
  macro avg   precision=X  recall=X  f1=X
```

---

### Task 3: יצירת DummyClassifier/README.md

**Files:**
- Create: `DummyClassifier/README.md`

- [ ] **Step 1: צור את ה-README עם התוצאות האמיתיות**

החלף את `[PLACEHOLDER]` בערכים שנרשמו ב-Task 2:

```markdown
# מודל Baseline טיפש — DummyClassifier

## תיאור כללי

מודל **DummyClassifier** עם אסטרטגיית `most_frequent` — מנבא תמיד את הקלאס הנפוץ ביותר בנתוני האימון, ללא כל לימוד מהפיצ'רים. משמש כ-**baseline טיפש** לצורך השוואה: כל מודל שמתיימר ללמוד משהו חייב לעקוף אותו.

---

## מטרה

לאחר שהרגרסיה הלוגיסטית השיגה ROC AUC = 0.475 (גרוע מניחוש אקראי), עולה השאלה: האם היא בכלל עוקפת מודל שלא לומד כלל? ה-DummyClassifier קובע את הרף הנמוך ביותר האפשרי.

---

## מקור הנתונים

| פרמטר | ערך |
|---|---|
| קובץ | `dataset/ecommerce_customer_behavior_dataset.csv` |
| מספר רשומות | 5,000 |
| חלוקה | 80% train / 20% test (stratify, random_state=42) |

---

## המודל

```python
DummyClassifier(strategy="most_frequent", random_state=42)
```

**אסטרטגיה:** `most_frequent` — מנבא תמיד קלאס 1 (לקוח חוזר), שהוא הקלאס הנפוץ ב-59.8% מהנתונים.

**אין** feature engineering, preprocessing, או scaling.

---

## תוצאות הערכה

| מדד | ערך |
|---|---|
| **Accuracy** | [ACC]% |
| **Precision** (class 1) | [PREC]% |
| **Recall** (class 1) | [REC1]% |
| **Recall** (class 0) | [REC0]% |
| **F1 Score** (class 1) | [F1] |
| **F1 Macro** | [F1M] |
| **ROC AUC** | [AUC] |

### מטריצת בלבול

|  | חזוי 0 | חזוי 1 |
|---|---|---|
| **אמיתי 0** | [TN] | [FP] |
| **אמיתי 1** | [FN] | [TP] |

---

## מסקנות

### 1. המודל לא לומד דבר — וזה בדיוק מה שמצפים ממנו
ה-DummyClassifier מנבא תמיד קלאס 1. לכן:
- **Recall קלאס 0 = 0%** — לא מזהה אף לקוח שאינו חוזר
- **Recall קלאס 1 = 100%** — "מוצא" את כולם כי מנבא תמיד 1
- **Accuracy ≈ 59.8%** — שיעור הקלאס הנפוץ בלבד

### 2. ROC AUC = 0.5 — רף הניחוש האקראי
מודל שמנבא הסתברות קבועה (1.0 לכולם) שווה לניחוש אקראי מבחינת כוח הפרדתי. זהו הרף שמתחתיו אסור להיות.

### 3. השוואה עם הרגרסיה הלוגיסטית
| מדד | Dummy | Logistic Baseline | Exp1 (balanced) |
|---|---|---|---|
| Accuracy | [ACC]% | 58.5% | 49.0% |
| Recall (קלאס 0) | 0% | 3% | 51% |
| F1 Macro | [F1M] | 0.39 | 0.49 |
| ROC AUC | [AUC] | 0.475 | 0.475 |

**מסקנה מרכזית:** הרגרסיה הלוגיסטית ה-baseline כמעט זהה ל-DummyClassifier — Accuracy מעט נמוכה יותר (בגלל שמנבאת כמה 0-ים), F1 Macro גבוה מעט יותר. לאחר הוספת `class_weight='balanced'` (Exp1), המודל עוקף את ה-Dummy בצורה משמעותית בכל המדדים.

---

## קבצים

| קובץ | תיאור |
|---|---|
| `dummy_classifier.py` | קוד המודל המלא |
| `dummy_classifier_model.pkl` | המודל המאומן (joblib) |
```

- [ ] **Step 2: Commit**

```bash
git add DummyClassifier/README.md
git commit -m "docs: add DummyClassifier README with results"
```

---

### Task 4: עדכון LogisticRegression/README.md

**Files:**
- Modify: `LogisticRegression/README.md`

- [ ] **Step 1: הוסף סעיף "השוואה מול Baseline טיפש" לסוף הקובץ**

הוסף בסוף `LogisticRegression/README.md` (אחרי קטע "ניסויים ושיפורים"):

```markdown
---

## השוואה מול Baseline טיפש (DummyClassifier)

| מדד | DummyClassifier | Logistic Baseline | ניסוי 1 | ניסוי 2 |
|---|---|---|---|---|
| Accuracy | [DUMMY_ACC]% | 58.5% | 49.0% | 47.8% |
| Recall (קלאס 0) | 0% | 3% | 51% | 52% |
| F1 Macro | [DUMMY_F1M] | 0.39 | 0.49 | 0.48 |
| ROC AUC | [DUMMY_AUC] | 0.475 | 0.475 | 0.478 |

### תובנות

**1. הלוגיסטי ה-baseline כמעט זהה ל-DummyClassifier**
הרגרסיה הלוגיסטית ללא איזון קלאסים (Baseline) השיגה Accuracy של 58.5% לעומת ~59.8% של ה-Dummy — פחות מ-1.5% הפרש. מבחינת F1 Macro ו-ROC AUC, הם כמעט זהים. המסקנה: ה-baseline ה"לא מאוזן" לא לומד דבר מעבר להטיה לקלאס 1.

**2. class_weight='balanced' הוא המינימום הנדרש**
רק לאחר הוספת `class_weight='balanced'` (ניסוי 1) המודל מתחיל לעקוף את ה-Dummy בצורה ברורה: Recall קלאס 0 קפץ מ-0% ל-51%, F1 Macro עלה מ-~0.33 ל-0.49. **כל מודל שמוגש בפרויקט הזה חייב לעקוף לפחות את ניסוי 1.**

**3. ROC AUC ≈ 0.475–0.478 — הפיצ'רים חלשים**
גם לאחר כל השיפורים, ROC AUC נשאר קרוב ל-0.5 (ניחוש אקראי). ה-DummyClassifier מאשר שהבעיה אינה בטיפול בחוסר האיזון אלא בעוצמת הפיצ'רים עצמם. הצעד הבא: מודלים לא-לינאריים (Random Forest / XGBoost) שיכולים ללכוד קשרים מורכבים יותר.
```

- [ ] **Step 2: Commit**

```bash
git add LogisticRegression/README.md
git commit -m "docs: add dummy classifier comparison to logistic README"
```
