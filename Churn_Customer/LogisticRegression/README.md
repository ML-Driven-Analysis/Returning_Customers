# Logistic Regression — ניבוי נטישת לקוחות (E-Commerce Customer Churn)

## מטרה

לבדוק האם Logistic Regression יכול לנבא נטישת לקוח בחנות אונליין, עם מתודולוגיה נכונה: **חלוקת Train/Test** (70%/30%, stratified) — קבוצת ה-Test נשמרת בצד ונבדקת **פעם אחת בלבד**, בסוף הניסוי. כל שלבי האימון, ה-Cross-Validation, ה-GridSearchCV וה-threshold tuning מתבצעים רק על קבוצת ה-Train; ה-Test משמש בלעדית למדידת ביצועים סופית בלתי-מוטה.

---

## נתונים

| פרמטר | ערך |
|---|---|
| קובץ | `E Commerce Dataset.xlsx` (גיליון: E Comm) |
| שורות | 5,630 |
| Target | `Churn` (1 = נטש, 0 = נשאר) |
| חוסר איזון | 83.2% לא נטשו (4,682) / 16.8% נטשו (948) — יחס 4.94:1 |
| Train | 3,941 שורות (664 נטשו, 16.8%) — 70% מהדאטה |
| Test (holdout) | 1,689 שורות (284 נטשו, 16.8%) — 30% מהדאטה, נבדק פעם אחת בסוף |

חלוקת ה-Train/Test בוצעה עם `stratify=y` כדי לשמר את אחוז הנטישה (16.8%) בשתי הקבוצות, ו-`random_state=42` לשחזוריות.

---

## ניסוי 1 — Baseline Logistic Regression

**קובץ:** `logistic_experiment_1_holdout.py`

### גישה

Pipeline בסיסי ללא feature engineering וללא טיפול בחוסר איזון — נקודת ייחוס נקייה:

```
Train/Test split: 70% / 30%, stratified, random_state=42

ColumnTransformer:
  numeric (13)    → SimpleImputer(median) + StandardScaler
  categorical (5) → SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown="ignore")
→ LogisticRegression(max_iter=1000, solver='lbfgs')

5-Fold StratifiedKFold (shuffle=True, random_state=42) על ה-TRAIN — לאימון ולבחירת המודל
בדיקה סופית על ה-TEST, פעם אחת, אחרי אימון על כל ה-TRAIN
```

### פיצ'רים

**Numeric (13):** Tenure, WarehouseToHome, HourSpendOnApp, NumberOfDeviceRegistered, SatisfactionScore, NumberOfAddress, Complain, OrderAmountHikeFromlastYear, CouponUsed, OrderCount, DaySinceLastOrder, CashbackAmount, CityTier

**Categorical (5):** PreferredLoginDevice, PreferredPaymentMode, Gender, PreferedOrderCat, MaritalStatus

### תוצאות (על קבוצת ה-Test, חד-פעמי)

| מדד | ערך |
|---|---|
| Accuracy | 89.17% |
| Balanced Accuracy | 73.82% |
| Precision (נוטשים) | 77.01% |
| Recall (נוטשים) | **50.70%** |
| F1 (נוטשים) | 61.15% |
| F1 Macro | 77.43% |
| ROC AUC | **88.36%** |
| PR AUC | 69.35% |

### מטריצת בלבול (Test)

|  | חזוי: נשאר | חזוי: נטש |
|---|---|---|
| **אמיתי: נשאר** | 1,362 | 43 |
| **אמיתי: נטש** | 140 | 144 |

### מסקנות

**✅ Signal אמיתי קיים:** ROC AUC של 88.4% על קבוצת test בלתי-נראית מצביע על קשר סטטיסטי משמעותי בין הפיצ'רים ל-label, גם במודל הבסיסי ביותר.

**⚠️ חולשה — Recall נמוך:** המודל מפספס 140 מתוך 284 נוטשים (49%) בקבוצת ה-test. זה נובע מחוסר האיזון: המודל "בוחר" לנבא "לא נטש" כברירת מחדל כי זה נכון ב-83% מהמקרים. בעולם העסקי זו בעיה קריטית — כל נוטש שלא זוהה הוא לקוח שאבד בלי התראה.

**📊 הדיוק (Accuracy) מטעה:** 89.2% נראה מרשים, אבל זה בעיקר תוצאה של חוסר האיזון. Balanced Accuracy (73.8%) הוא המדד האמיתי.

---

## ניסוי 2 — class_weight + Feature Engineering + Threshold Tuning

**קובץ:** `logistic_experiment_2_holdout.py`

### גישה

שלושה שיפורים על ניסוי 1, כולם מבוססים רק על קבוצת ה-Train:

```
Train/Test split: 70% / 30%, stratified, random_state=42
Feature Engineering: +4 פיצ'רים חדשים (16 numeric, 6 categorical)
GridSearchCV (TRAIN, 3-Fold): C in [0.01, 0.1, 1, 10, 100] → best C=1
LogisticRegression(class_weight='balanced', C=1)
Threshold tuning (TRAIN, Precision-Recall curve): optimal = 0.7365
בדיקה סופית — TEST, בשני הספים (0.5 ו-0.7365), פעם אחת
```

### פיצ'רים חדשים

| פיצ'ר | חישוב | משמעות |
|---|---|---|
| `AvgCashbackPerOrder` | `CashbackAmount / OrderCount` | קאשבק ממוצע לפי הזמנה |
| `IsHighComplainer` | `Complain == 1` | הגיש תלונה |
| `LowSatisfaction` | `SatisfactionScore <= 2` | שביעות רצון נמוכה מאוד |
| `DaysSinceOrderBucket` | recent (≤7 ימים) / medium (≤30) / long | גיל ההזמנה האחרונה |

### GridSearchCV (TRAIN, 3-Fold, scoring=ROC AUC)

| C | ROC AUC | סטיית תקן |
|---|---|---|
| 0.01 | 88.69% | ±0.96% |
| **1** | **88.95%** | ±1.10% |
| 10 | 88.94% | ±1.08% |
| 100 | 88.93% | ±1.08% |

> C=1 נבחר — ההבדל מ-C=10 זניח, ופחות רגולריזציה לא מוסיפה ערך אבל מעלה חשש ל-overfitting.

### תוצאות — Threshold = 0.5 (על קבוצת ה-Test, חד-פעמי)

| מדד | ערך |
|---|---|
| Accuracy | 79.51% |
| Balanced Accuracy | 80.52% |
| Precision (נוטשים) | 44.13% |
| **Recall (נוטשים)** | **82.04%** |
| F1 (נוטשים) | 57.39% |
| F1 Macro | 71.95% |
| ROC AUC | 88.45% |

### תוצאות — Threshold = 0.7365 (אופטימלי, נמצא על TRAIN; נבדק על Test)

| מדד | ערך |
|---|---|
| Accuracy | 87.27% |
| Balanced Accuracy | 77.46% |
| Precision (נוטשים) | 62.02% |
| **Recall (נוטשים)** | **62.68%** |
| F1 (נוטשים) | 62.35% |
| **F1 Macro** | **77.34%** |
| ROC AUC | 88.45% |

### מטריצות בלבול (Test)

**Threshold = 0.5:**

|  | חזוי: נשאר | חזוי: נטש |
|---|---|---|
| **אמיתי: נשאר** | 1,110 | 295 |
| **אמיתי: נטש** | 51 | 233 |

**Threshold = 0.7365:**

|  | חזוי: נשאר | חזוי: נטש |
|---|---|---|
| **אמיתי: נשאר** | 1,296 | 109 |
| **אמיתי: נטש** | 106 | 178 |

### מסקנות

**threshold=0.5 — הכי הרבה נוטשים נתפסים:** Recall של 82.0% (233 מתוך 284 נוטשים נתפסו) — אך Precision נמוך (44.1%), כלומר יותר ממחצית מהתראות הנטישה שגויות.

**threshold=0.7365 — האיזון הטוב ביותר:** F1 Macro מקסימלי (77.3%), Precision ו-Recall מאוזנים (62%/63%).

---

## השוואה סופית (כל המדדים על Test בלבד)

| מדד | ניסוי 1 (baseline) | ניסוי 2 (threshold=0.5) | ניסוי 2 (threshold=0.7365) |
|---|---|---|---|
| ROC AUC | **88.4%** | 88.5% | 88.5% |
| Recall (נוטשים) | 50.7% | **82.0%** | 62.7% |
| Precision (נוטשים) | **77.0%** | 44.1% | 62.0% |
| F1 Macro | 77.4% | 72.0% | **77.3%** |
| Balanced Accuracy | 73.8% | **80.5%** | 77.5% |

### ROC AUC — כמעט ללא שינוי (~88.5%)
Feature engineering לא הוסיף signal חדש שמשנה את יכולת ההפרדה הבסיסית של המודל — ה-AUC עמיד לסף ולמשקלות, ומייצג את "תקרת" הביצועים של הגישה הזו.

### class_weight='balanced' עם threshold=0.5 — הכי הרבה נוטשים נתפסים
Recall קפץ מ-50.7% (ניסוי 1) ל-82.0%. המחיר: Precision צנח ל-44.1% — כמעט מחצית מהתראות ה"נטישה" שגויות. נכון לבחור כשעלות פנייה לא-נוטש זולה מעלות אובדן נוטש.

### threshold=0.7365 — האיזון הטוב ביותר לשימוש כללי
F1 Macro מקסימלי, Precision ו-Recall קרובים זה לזה.

### הבחירה תלויה בהחלטה עסקית

| מצב | Threshold מומלץ |
|---|---|
| קמפיין שימור — כל פנייה עולה כסף | 0.7365 (מאוזן) |
| פנייה זולה (אימייל/פוש) — עדיף לא לפספס נוטש | 0.5 (Recall גבוה) |

---

## מבנה הקוד והערכה משותפת

שני הניסויים משתמשים בקובץ משותף `../evaluation_utils.py` שמכיל:
- `evaluate_model_cv(model, X, y, cv, thresholds)` — מריץ Cross-Validation על ה-Train ומחזיר מדדים אחידים (Accuracy, Balanced Accuracy, Precision/Recall/F1 למחלקת הנוטשים, F1 Macro, ROC AUC, PR AUC, Confusion Matrix) לכל סף שמתבקש.
- `evaluate_predictions(y, y_proba, thresholds)` — אותם מדדים, על תחזיות שכבר חושבו — משמש לבדיקת ה-Test הסופית החד-פעמית.
- `print_evaluation(results, label)` — הדפסה אחידה של התוצאות.

כך מובטח שכל המודלים בפרויקט (Logistic Regression, XGBoost) נמדדים באותה שיטה ובאותם מדדים.

---

## קבצים

| קובץ | תיאור |
|---|---|
| `logistic_experiment_1_holdout.py` | ניסוי 1 — baseline + Train/Test split |
| `logistic_experiment_1_holdout_summary.csv` | תוצאות ניסוי 1 |
| `logistic_experiment_2_holdout.py` | ניסוי 2 — class_weight + feature engineering + threshold tuning + Train/Test split |
| `logistic_experiment_2_holdout_summary.csv` | תוצאות ניסוי 2 (default + tuned) |
| `../evaluation_utils.py` | פונקציות הערכה משותפות לכל המודלים בפרויקט |
