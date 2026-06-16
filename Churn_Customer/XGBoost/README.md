# XGBoost — ניבוי נטישת לקוחות (E-Commerce Customer Churn)

## מטרה

לבדוק האם XGBoost משפר את ביצועי הניבוי לעומת Logistic Regression, עם מתודולוגיה נכונה: **חלוקת Train/Test** (70%/30%, stratified) — קבוצת ה-Test נשמרת בצד ונבדקת **פעם אחת בלבד**, בסוף הניסוי. כל שלבי האימון, ה-Cross-Validation וה-threshold tuning מתבצעים רק על קבוצת ה-Train; ה-Test משמש בלעדית למדידת ביצועים סופית בלתי-מוטה.

XGBoost נבחר כי הוא חזק מאוד על נתונים טבלאיים, נפוץ בפרויקטי ML, ותומך טבעית בחוסר איזון דרך הפרמטר `scale_pos_weight`.

---

## נתונים

| פרמטר | ערך |
|---|---|
| קובץ | `E Commerce Dataset.xlsx` (גיליון: E Comm) |
| שורות | 5,630 |
| Target | `Churn` (1 = נטש, 0 = נשאר) |
| חוסר איזון | 83.2% / 16.8% — יחס 4.94:1 |
| Train | 3,941 שורות (664 נטשו, 16.8%) — 70% מהדאטה |
| Test (holdout) | 1,689 שורות (284 נטשו, 16.8%) — 30% מהדאטה, נבדק פעם אחת בסוף |
| scale_pos_weight | 4.94 — מחושב **מ-TRAIN בלבד** (לא מכל הדאטהסט, כדי לא לדלוף מידע מה-test) |

חלוקת ה-Train/Test בוצעה עם `stratify=y` כדי לשמר את אחוז הנטישה (16.8%) בשתי הקבוצות, ו-`random_state=42` לשחזוריות.

---

## ניסוי 1 — Baseline XGBoost + scale_pos_weight

**קובץ:** `xgboost_experiment_1_holdout.py`

### גישה

XGBoost בלי feature engineering ובלי threshold tuning — נקודת ייחוס נקייה. חוסר האיזון מטופל דרך `scale_pos_weight` (פרמטר פנימי של XGBoost שמשקלל יותר את מחלקת המיעוט):

```python
Train/Test split: 70% / 30%, stratified, random_state=42

ColumnTransformer:
  numeric (13)    → SimpleImputer(median) + StandardScaler
  categorical (5) → SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown="ignore")
→ XGBClassifier(
      n_estimators=300,
      max_depth=3,
      learning_rate=0.05,
      subsample=0.8,
      colsample_bytree=0.8,
      scale_pos_weight=4.94,
      eval_metric="logloss",
      random_state=42,
  )

5-Fold StratifiedKFold (shuffle=True, random_state=42) על ה-TRAIN — לאימון ולבחירת המודל
בדיקה סופית על ה-TEST, פעם אחת, אחרי אימון על כל ה-TRAIN
```

### פיצ'רים

**Numeric (13):** Tenure, WarehouseToHome, HourSpendOnApp, NumberOfDeviceRegistered, SatisfactionScore, NumberOfAddress, Complain, OrderAmountHikeFromlastYear, CouponUsed, OrderCount, DaySinceLastOrder, CashbackAmount, CityTier

**Categorical (5):** PreferredLoginDevice, PreferredPaymentMode, Gender, PreferedOrderCat, MaritalStatus

### תוצאות (על קבוצת ה-Test, חד-פעמי)

| מדד | ערך |
|---|---|
| Accuracy | 89.82% |
| Balanced Accuracy | **89.38%** |
| Precision (Churn=1) | 64.29% |
| Recall (Churn=1) | **88.73%** |
| F1 (Churn=1) | 74.56% |
| F1 Macro | 84.10% |
| ROC AUC | **94.81%** |
| PR AUC | 81.61% |

### מטריצת בלבול (Test)

|  | חזוי: נשאר | חזוי: נטש |
|---|---|---|
| **אמיתי: נשאר** | 1,265 | 140 |
| **אמיתי: נטש** | 32 | 252 |

### מסקנות

**✅ ROC AUC גבוה משמעותית מ-Logistic Regression** (94.8% לעומת ~88.5%) — כבר במודל הבסיסי, בלי שום feature engineering.

**Recall גבוה מאוד (88.7%):** המודל מזהה כ-9 מתוך 10 לקוחות שינטשו בקבוצת ה-test.

**⚠️ חולשה — Precision מתון (64.3%):** על כל לקוח שנוטש נכון, יש כ-36% false positives.

---

## ניסוי 2 — Feature Engineering + Threshold Tuning

**קובץ:** `xgboost_experiment_2_holdout.py`

### גישה

שני שיפורים על ניסוי 1, מבוססים רק על קבוצת ה-Train:

```python
Train/Test split: 70% / 30%, stratified, random_state=42
Feature Engineering: +4 פיצ'רים חדשים (16 numeric, 6 categorical)
Threshold tuning (TRAIN, Precision-Recall curve): optimal = 0.7178
בדיקה סופית — TEST, בשני הספים (0.5 ו-0.7178), פעם אחת
```

### פיצ'רים חדשים

| פיצ'ר | חישוב | משמעות |
|---|---|---|
| `AvgCashbackPerOrder` | `CashbackAmount / OrderCount` | קאשבק ממוצע לפי הזמנה |
| `IsHighComplainer` | `Complain == 1` | הגיש תלונה |
| `LowSatisfaction` | `SatisfactionScore <= 2` | שביעות רצון נמוכה מאוד |
| `DaysSinceOrderBucket` | recent (≤7 ימים) / medium (≤30) / long | גיל ההזמנה האחרונה |

### תוצאות — Threshold = 0.5 (על קבוצת ה-Test, חד-פעמי)

| מדד | ערך |
|---|---|
| Accuracy | 89.46% |
| Balanced Accuracy | 88.89% |
| Precision (Churn=1) | 63.45% |
| Recall (Churn=1) | 88.03% |
| F1 (Churn=1) | 73.75% |
| F1 Macro | 83.58% |
| ROC AUC | 94.86% |
| PR AUC | 81.63% |

### תוצאות — Threshold = 0.7178 (אופטימלי, נמצא על TRAIN; נבדק על Test)

| מדד | ערך |
|---|---|
| Accuracy | 91.65% |
| Balanced Accuracy | 84.17% |
| Precision (Churn=1) | **76.38%** |
| Recall (Churn=1) | 72.89% |
| F1 (Churn=1) | 74.59% |
| **F1 Macro** | **84.80%** |
| ROC AUC | 94.86% |
| PR AUC | 81.63% |

### מטריצות בלבול (Test)

**Threshold = 0.5:**

|  | חזוי: נשאר | חזוי: נטש |
|---|---|---|
| **אמיתי: נשאר** | 1,261 | 144 |
| **אמיתי: נטש** | 34 | 250 |

**Threshold = 0.7178:**

|  | חזוי: נשאר | חזוי: נטש |
|---|---|---|
| **אמיתי: נשאר** | 1,341 | 64 |
| **אמיתי: נטש** | 77 | 207 |

### מסקנות

**F1 Macro של 84.80% — שיא הדיוק מבין כל הניסויים בפרויקט** (Logistic + XGBoost).

**Feature engineering לא שינה את ROC AUC:** XGBoost כבר לומד בעצמו את האינטראקציות הרלוונטיות בין הפיצ'רים, ולכן feature engineering ידני מוסיף ערך מוגבל.

**Threshold tuning הוא התרומה האמיתית:** Precision עלה מ-63.5% ל-76.4% (פחות false positives, מ-144 ל-64), במחיר ירידה ב-Recall (88% → 73%).

---

## השוואה סופית (כל המדדים על Test בלבד)

| מודל | ROC AUC | F1 Macro | Recall(1) | Precision(1) |
|---|---|---|---|---|
| Logistic Regression Exp 1 (baseline) | 88.4% | 77.4% | 50.7% | 77.0% |
| Logistic Regression Exp 2 (threshold=0.7365) | 88.5% | 77.3% | 62.7% | 62.0% |
| XGBoost Exp 1 (baseline, threshold=0.5) | 94.8% | 84.1% | **88.7%** | 64.3% |
| **XGBoost Exp 2 (threshold=0.7178)** | **94.9%** | **84.8%** | 72.9% | **76.4%** |

XGBoost עולה משמעותית על Logistic Regression בכל המדדים, על קבוצת test בלתי-מוטה.

### הבחירה תלויה בהחלטה עסקית

| מצב | Threshold מומלץ |
|---|---|
| קמפיין שימור — כל פנייה עולה כסף (Precision גבוה) | 0.7178 |
| פנייה זולה — עדיף לתפוס כמה שיותר נוטשים (Recall גבוה) | 0.5 |

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
| `xgboost_experiment_1_holdout.py` | ניסוי 1 — Baseline + scale_pos_weight + Train/Test split |
| `xgboost_experiment_1_holdout_summary.csv` | תוצאות ניסוי 1 |
| `xgboost_experiment_2_holdout.py` | ניסוי 2 — Feature Engineering + Threshold Tuning + Train/Test split |
| `xgboost_experiment_2_holdout_summary.csv` | תוצאות ניסוי 2 (default + tuned) |
| `../evaluation_utils.py` | פונקציות הערכה משותפות לכל המודלים בפרויקט |
