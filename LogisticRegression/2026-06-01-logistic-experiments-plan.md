# Logistic Regression Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** לבצע שני ניסויים מצטברים לשיפור המודל הלוגיסטי ולתעד את התוצאות ב-README.

**Architecture:** כל ניסוי הוא קובץ Python עצמאי שנגזר מה-baseline. ניסוי 1 מוסיף `class_weight='balanced'`. ניסוי 2 מצטבר עליו ומסיר 5 פיצ'רים מקורלים. לאחר כל הרצה מעדכנים את README עם תוצאות אמיתיות.

**Tech Stack:** Python, pandas, scikit-learn, joblib

---

### Task 1: יצירת logistic_experiment_1.py

**Files:**
- Create: `LogisticRegression/logistic_experiment_1.py`

- [ ] **Step 1: העתק את קובץ ה-baseline**

```bash
cp LogisticRegression/logistic_regression.py LogisticRegression/logistic_experiment_1.py
```

- [ ] **Step 2: שנה את שם קובץ המודל הנשמר (שורה 366)**

מצא:
```python
MODEL_FILE = BASE_DIR / "logistic_regression_returning_customer_model.pkl"
```
החלף ב:
```python
MODEL_FILE = BASE_DIR / "logistic_experiment_1_model.pkl"
```

- [ ] **Step 3: הוסף `class_weight='balanced'` (שורות 274-277)**

מצא:
```python
    ("classifier", LogisticRegression(
        max_iter=3000,
        random_state=42
    ))
```
החלף ב:
```python
    ("classifier", LogisticRegression(
        max_iter=3000,
        random_state=42,
        class_weight="balanced"
    ))
```

- [ ] **Step 4: Commit**

```bash
git add LogisticRegression/logistic_experiment_1.py
git commit -m "feat: add experiment 1 - class_weight=balanced"
```

---

### Task 2: הרצת ניסוי 1 ובדיקת קריטריוני הצלחה

**Files:**
- Run: `LogisticRegression/logistic_experiment_1.py`

- [ ] **Step 1: הרץ את הסקריפט**

```bash
python LogisticRegression/logistic_experiment_1.py
```

- [ ] **Step 2: בדוק קריטריוני הצלחה בפלט**

מצא בפלט את הערכים הבאים ורשום אותם (יידרשו ב-Task 5):

```
Accuracy: X.XXX
ROC AUC: X.XXX          ← חייב להיות > 0.5
Confusion Matrix:
[[ TN  FP ]
 [ FN  TP ]]
Classification Report:
              precision    recall  f1-score
           0       X.XX      X.XX      X.XX   ← Recall של קלאס 0 חייב להיות > 0.30
           1       X.XX      X.XX      X.XX
    macro avg      X.XX      X.XX      X.XX   ← F1 Macro לרשום
```

אם ROC AUC ≤ 0.5 או Recall קלאס 0 ≤ 0.10 — בדוק שהשינוי ב-Step 3 של Task 1 בוצע נכון.

---

### Task 3: יצירת logistic_experiment_2.py

**Files:**
- Create: `LogisticRegression/logistic_experiment_2.py`

- [ ] **Step 1: העתק את ניסוי 1**

```bash
cp LogisticRegression/logistic_experiment_1.py LogisticRegression/logistic_experiment_2.py
```

- [ ] **Step 2: שנה את שם קובץ המודל הנשמר**

מצא:
```python
MODEL_FILE = BASE_DIR / "logistic_experiment_1_model.pkl"
```
החלף ב:
```python
MODEL_FILE = BASE_DIR / "logistic_experiment_2_model.pkl"
```

- [ ] **Step 3: הוסף פיצ'רים לרשימת columns_to_drop (סעיף 5, שורות 208-221)**

מצא:
```python
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
```
החלף ב:
```python
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
    # Experiment 2: remove redundant binary features (continuous versions already present)
    "Original_Amount",      # = Unit_Price * Quantity, correlated with Total_Amount
    "Long_Session",         # binary version of Session_Duration_Minutes
    "Many_Pages_Viewed",    # binary version of Pages_Viewed
    "Fast_Delivery",        # binary version of Delivery_Time_Days
    "High_Rating",          # binary version of Customer_Rating
]
```

- [ ] **Step 4: Commit**

```bash
git add LogisticRegression/logistic_experiment_2.py
git commit -m "feat: add experiment 2 - remove correlated features"
```

---

### Task 4: הרצת ניסוי 2 ובדיקת קריטריוני הצלחה

**Files:**
- Run: `LogisticRegression/logistic_experiment_2.py`

- [ ] **Step 1: הרץ את הסקריפט**

```bash
python LogisticRegression/logistic_experiment_2.py
```

- [ ] **Step 2: בדוק שהפיצ'רים הוסרו בפלט**

חפש בפלט:
```
Features used by the model:
```
וודא שהרשימה **לא כוללת**: `Original_Amount`, `Long_Session`, `Many_Pages_Viewed`, `Fast_Delivery`, `High_Rating`.

- [ ] **Step 3: רשום את תוצאות ניסוי 2**

```
Accuracy: X.XXX
ROC AUC: X.XXX          ← צפוי להיות > ניסוי 1 או דומה
Confusion Matrix:
[[ TN  FP ]
 [ FN  TP ]]
Classification Report:
              precision    recall  f1-score
           0       X.XX      X.XX      X.XX
           1       X.XX      X.XX      X.XX
    macro avg      X.XX      X.XX      X.XX   ← F1 Macro חייב להיות > 0.45
```

---

### Task 5: עדכון README עם תוצאות אמיתיות

**Files:**
- Modify: `LogisticRegression/README.md`

- [ ] **Step 1: הוסף קטע "ניסויים ושיפורים" לסוף ה-README**

החלף את `[ערכי_EXP1]` ו-`[ערכי_EXP2]` בתוצאות שנרשמו ב-Task 2 ו-Task 4:

```markdown
---

## ניסויים ושיפורים

### טבלת השוואה מרכזית

| מדד | Baseline | ניסוי 1 | ניסוי 2 |
|---|---|---|---|
| Accuracy | 58.5% | [EXP1_ACC]% | [EXP2_ACC]% |
| Recall (קלאס 0) | 3% | [EXP1_REC0]% | [EXP2_REC0]% |
| Recall (קלאס 1) | 95.8% | [EXP1_REC1]% | [EXP2_REC1]% |
| F1 Macro | 0.39 | [EXP1_F1M] | [EXP2_F1M] |
| ROC AUC | 0.475 | [EXP1_AUC] | [EXP2_AUC] |

---

### ניסוי 1: class_weight='balanced'

**קובץ:** `logistic_experiment_1.py`

**שינוי שבוצע:**
```python
LogisticRegression(max_iter=3000, random_state=42, class_weight="balanced")
```

**מטריצת בלבול:**

|  | חזוי 0 | חזוי 1 |
|---|---|---|
| **אמיתי 0** | [EXP1_TN] | [EXP1_FP] |
| **אמיתי 1** | [EXP1_FN] | [EXP1_TP] |

**מסקנה:** [מלא לאחר הרצה — האם ROC AUC עלה מעל 0.5? האם המודל מזהה כעת לקוחות לא-חוזרים?]

---

### ניסוי 2: class_weight='balanced' + הסרת פיצ'רים מקורלים

**קובץ:** `logistic_experiment_2.py`

**שינויים שבוצעו:**
- `class_weight='balanced'` (מניסוי 1)
- הוסרו: `Original_Amount`, `Long_Session`, `Many_Pages_Viewed`, `Fast_Delivery`, `High_Rating`

**מטריצת בלבול:**

|  | חזוי 0 | חזוי 1 |
|---|---|---|
| **אמיתי 0** | [EXP2_TN] | [EXP2_FP] |
| **אמיתי 1** | [EXP2_FN] | [EXP2_TP] |

**מסקנה:** [מלא לאחר הרצה — האם ניקוי הפיצ'רים שיפר את המדדים מעבר לניסוי 1?]
```

- [ ] **Step 2: Commit**

```bash
git add LogisticRegression/README.md
git commit -m "docs: add experiment results to README"
```
