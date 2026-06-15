# Churn Customer — ניסויי CatBoost

---

##  מטרה

חיזוי האם לקוח ינטוש (Churn=1) או יישאר (Churn=0) בפלטפורמת מסחר אלקטרוני,
על בסיס נתוני התנהגות, העדפות ושביעות רצון.

---

##  נתונים

| פרמטר | ערך |
|-------|-----|
| קובץ | `E Commerce Dataset.xlsx` (Sheet: E Comm) |
| שורות | 5,630 |
| עמודות | 20 |
| משתנה מטרה | `Churn` (1 = נטש, 0 = נשאר) |
| התפלגות | 4,682 נשארו (83.2%) / 948 נטשו (16.8%) |
| חוסר איזון | יחס 4.94:1 לטובת המחלקה השלילית |

---

##  ניסוי 1 — CatBoost Baseline

### גישה

- מודל CatBoost עם פרמטרים ברירת מחדל
- אימפוטציה: חציון לנומריים, ערך שכיח לקטגוריים
- ללא תיקון חוסר איזון
- סף קלסיפיקציה: 0.50
- ולידציה: StratifiedKFold (5 folds, random_state=42)

### פיצ'רים

**Numeric (13):**
Tenure, WarehouseToHome, HourSpendOnApp, NumberOfDeviceRegistered, SatisfactionScore,
NumberOfAddress, Complain, OrderAmountHikeFromlastYear, CouponUsed, OrderCount,
DaySinceLastOrder, CashbackAmount, CityTier

**Categorical (5):**
PreferredLoginDevice, PreferredPaymentMode, Gender, PreferedOrderCat, MaritalStatus

### תוצאות

| מדד | ערך |
|-----|-----|
| Accuracy | 0.9519 |
| Balanced Accuracy | 0.8815 |
| Precision (נוטשים) | 0.9269 |
| Recall (נוטשים) | 0.7753 |
| F1 (נוטשים) | 0.8443 |
| F1 Macro | 0.9079 |
| ROC AUC | 0.9835 |
| PR AUC | 0.9351 |

### מטריצת בלבול

```
                  Predicted 0    Predicted 1
  Actual 0  :       4,624            58
  Actual 1  :         213           735
```

---

##  מסקנות ניסוי 1

**הצלחה:**
- ROC AUC של 0.9835 — המודל מפריד היטב בין המחלקות
- Precision גבוה (0.927) — כשהמודל מנבא נטישה הוא צודק ב-93% מהמקרים
- Accuracy כללי גבוה מאוד (95.2%)

**חולשה:**
- Recall של 0.775 — המודל **מפספס 213 לקוחות נוטשים** (False Negatives)
- חוסר האיזון בדאטה (4.94:1) גורם למודל להטות לצד המחלקה הרוב
- מבחינה עסקית: לקוח נוטש שמתפספס עולה יותר מלקוח שמסווג בטעות כנוטש

---

##  ניסוי 2 — CatBoost עם Class Weights + Threshold Tuning

### גישה

שני שינויים מרכזיים ביחס לניסוי 1:

1. **`auto_class_weights="Balanced"`** — CatBoost מחשב אוטומטית משקל גבוה יותר למחלקת הנוטשים (1) בזמן האימון, כדי לפצות על חוסר האיזון
2. **Threshold Tuning** — בדיקת שלושה ספי קלסיפיקציה: 0.30, 0.40, 0.50, כדי לשלוט ב-tradeoff בין Recall ל-Precision

### פיצ'רים

זהים לניסוי 1 — ללא שינוי בפיצ'רים.

---

### תוצאות — Threshold = 0.50

| מדד | ערך |
|-----|-----|
| Accuracy | 0.9545 |
| Balanced Accuracy | 0.9386 |
| Precision (נוטשים) | 0.8321 |
| Recall (נוטשים) | 0.9146 |
| F1 (נוטשים) | 0.8714 |
| F1 Macro | 0.9219 |
| ROC AUC | 0.9859 |
| PR AUC | 0.9422 |

---

### תוצאות — Threshold = 0.40

| מדד | ערך |
|-----|-----|
| Accuracy | 0.9433 |
| Balanced Accuracy | 0.9411 |
| Precision (נוטשים) | 0.7737 |
| Recall (נוטשים) | 0.9378 |
| F1 (נוטשים) | 0.8479 |
| F1 Macro | 0.9065 |
| ROC AUC | 0.9859 |
| PR AUC | 0.9422 |

---

### תוצאות — Threshold = 0.30

| מדד | ערך |
|-----|-----|
| Accuracy | 0.9227 |
| Balanced Accuracy | 0.9359 |
| Precision (נוטשים) | 0.6975 |
| Recall (נוטשים) | 0.9557 |
| F1 (נוטשים) | 0.8064 |
| F1 Macro | 0.8791 |
| ROC AUC | 0.9859 |
| PR AUC | 0.9422 |

---

### מטריצות בלבול

**Threshold = 0.50**
```
                  Predicted 0    Predicted 1
  Actual 0  :       4,507           175
  Actual 1  :          81           867
```

**Threshold = 0.40**
```
                  Predicted 0    Predicted 1
  Actual 0  :       4,422           260
  Actual 1  :          59           889
```

**Threshold = 0.30**
```
                  Predicted 0    Predicted 1
  Actual 0  :       4,289           393
  Actual 1  :          42           906
```

---

##  השוואה סופית

| מדד | ניסוי 1 | ניסוי 2 (t=0.40) | ניסוי 2 (t=0.30) |
|-----|---------|-----------------|-----------------|
| ROC AUC | 0.9835 | 0.9859 ✅ | 0.9859 ✅ |
| Recall (נוטשים) | 0.7753 | 0.9378 ✅ | 0.9557 ✅ |
| Precision (נוטשים) | 0.9269 | 0.7737 ⚠️ | 0.6975 ⚠️ |
| F1 (נוטשים) | 0.8443 | 0.8479 ✅ | 0.8064 |
| נוטשים שהתפספסו (FN) | 213 | 59 ✅ | 42 ✅ |

---

## מסקנות

- הוספת `auto_class_weights="Balanced"` לבדה שיפרה את ה-Recall מ-77.5% ל-91.5% (threshold=0.50), ללא צורך בשינוי הדאטה
- ה-ROC AUC עלה מ-0.9835 ל-0.9859 — המודל השתפר גם ברמת הדירוג הכללי
- **Threshold = 0.40** מהווה את נקודת האיזון הטובה ביותר: Recall של 93.8% תוך שמירה על F1 גבוה ו-59 נוטשים בלבד שמתפספסים (לעומת 213 בניסוי 1)
- **Threshold = 0.30** מתאים כשהעדיפות העסקית היא לתפוס כמה שיותר נוטשים — Recall של 95.6%, אך על חשבון יותר False Positives (393 לקוחות שמסווגים בטעות כנוטשים)
- הפחתת ה-False Negatives מ-213 ל-59 (threshold=0.40) מייצגת שיפור של **72%** בתפיסת לקוחות נוטשים

---

##  קבצים

```
Churn_Customer/
├── dataset/
│   └── E Commerce Dataset.xlsx
├── experiments/
│   ├── catboost_experiment_1.py
│   ├── catboost_experiment_1_summary.csv
│   ├── catboost_experiment_2.py
│   ├── catboost_experiment_2_summary.csv
│   └── README.md
```