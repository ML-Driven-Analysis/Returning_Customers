# מודל XGBoost – חיזוי לקוחות חוזרים

## תיאור כללי

מודל **XGBoost (Extreme Gradient Boosting)** לחיזוי האם לקוח ב-e-commerce יהיה **לקוח חוזר**. זהו מודל עצי-החלטה לא-לינארי המסוגל ללכוד קשרים מורכבים בין פיצ'רים — בניגוד לרגרסיה לוגיסטית שהוגבלה ל-ROC AUC ≈ 0.49.

---

## מקור הנתונים

| פרמטר | ערך |
|---|---|
| קובץ | `dataset/ecommerce_customer_behavior_dataset.csv` |
| מספר רשומות | 5,000 |
| מספר עמודות מקוריות | 18 |
| ערכים חסרים | אין |

---

## תווית היעד

| קלאס | מספר | אחוז |
|---|---|---|
| 1 (לקוח חוזר) | 2,990 | 59.8% |
| 0 (לא חוזר) | 2,010 | 40.2% |

קיים **חוסר איזון מתון** לטובת לקוחות חוזרים. ניסוי 3 יטפל בכך עם `scale_pos_weight`.

---

## הנדסת פיצ'רים

מעבר לעמודות המקור, נוספו הפיצ'רים הבאים:

### פיצ'רי תאריך
- `Year`, `Month`, `DayOfWeek`, `DayOfMonth`, `Quarter` – פירוק התאריך
- `IsWeekend` – האם ההזמנה בסוף שבוע (0/1)

### פיצ'רי סכום
- `Original_Amount` = `Unit_Price × Quantity`
- `Has_Discount` – האם קיימת הנחה (0/1)
- `Discount_Rate` – שיעור ההנחה
- `Amount_Per_Item` – עלות ממוצעת לפריט

### פיצ'רי גלישה
- `Pages_Per_Minute` – קצב גלישה
- `Long_Session` – האם הסשן ארוך מהחציון (0/1)
- `Many_Pages_Viewed` – האם מספר העמודים גדול מהחציון (0/1)

### פיצ'רי משלוח ושביעות רצון
- `Fast_Delivery` – האם המשלוח מהיר מהחציון (0/1)
- `High_Rating` – האם הדירוג ≥ 4 (0/1)

**סה"כ פיצ'רים לפני One-Hot Encoding: 29**

> חציונים מחושבים אך ורק על נתוני אימון (בתוך ה-Transformer) — ללא data leakage.

---

## עיבוד מקדים

| סוג עמודה | טיפול |
|---|---|
| נומרי | `SimpleImputer(median)` בלבד — **ללא StandardScaler** |
| קטגוריאלי | `SimpleImputer(most_frequent)` → `OneHotEncoder` |

**למה אין StandardScaler?** XGBoost מבוסס עצי-החלטה — לא מושפע מסקאלת הפיצ'רים.

---

## ניסוי 1 — הגדרת XGBoost

| פרמטר | ערך |
|---|---|
| `n_estimators` | 100 |
| `max_depth` | 6 |
| `learning_rate` | 0.1 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `eval_metric` | logloss |
| `random_state` | 42 |
| `scale_pos_weight` | לא מוגדר (ניסוי baseline) |

---

## מדדי הערכה

הערכה מבוצעת באמצעות **StratifiedKFold (5 קפלים)** עם המדדים הבאים:

- Accuracy, Balanced Accuracy
- Precision / Recall / F1 לכל קלאס (0 ו-1)
- F1 Macro
- ROC AUC

---

## תוצאות — ניסוי 1

### ממוצעי 5 קפלים

| מדד | ממוצע | סטיית תקן |
|---|---|---|
| Accuracy | 54.6% | ±1.05% |
| Balanced Accuracy | 48.7% | ±0.88% |
| Precision (קלאס 0) | 37.0% | ±2.14% |
| Recall (קלאס 0) | 18.5% | ±2.10% |
| F1 (קלאס 0) | 0.246 | ±0.020 |
| Precision (קלאס 1) | 59.0% | ±0.53% |
| Recall (קלאס 1) | 78.9% | ±2.45% |
| F1 (קלאס 1) | 0.675 | ±0.012 |
| **F1 Macro** | **0.460** | ±0.010 |
| **ROC AUC** | **0.489** | ±0.003 |

### מטריצת בלבול (OOF מצטברת — 5,000 רשומות)

|  | חזוי 0 | חזוי 1 |
|---|---|---|
| **אמיתי 0** | 371 | 1,639 |
| **אמיתי 1** | 632 | 2,358 |

### Top 15 Feature Importances

| פיצ'ר | חשיבות |
|---|---|
| `City_Kayseri` | 0.0240 |
| `City_Bursa` | 0.0239 |
| `Payment_Cash on Delivery` | 0.0233 |
| `City_Eskisehir` | 0.0233 |
| `Original_Amount` | 0.0231 |
| `Product_Home & Garden` | 0.0228 |
| `City_Izmir` | 0.0225 |
| `Product_Sports` | 0.0224 |
| `City_Ankara` | 0.0219 |
| `Payment_Bank Transfer` | 0.0216 |
| `City_Adana` | 0.0213 |
| `Product_Fashion` | 0.0212 |
| `Amount_Per_Item` | 0.0207 |
| `Gender_Other` | 0.0206 |
| `City_Gaziantep` | 0.0202 |

---

## מסקנות — ניסוי 1

### 1. XGBoost עדיף על Logistic Baseline — אך עדיין לא מאוזן
F1 Macro של **0.460** גבוה מהרגרסיה הלוגיסטית ה-baseline (0.396), אבל Recall קלאס 0 הוא **18.5% בלבד** — המודל עדיין מוטה לחזות קלאס 1.

### 2. חשיבות פיצ'רים שטוחה — אין פיצ'ר מוביל
כל 15 הפיצ'רים המובילים מקבלים חשיבות 0.020–0.024 — פיזור שווה שמצביע על כך שאין קשר חזק בין שום פיצ'ר לתווית. זה עקבי עם ממצאי הרגרסיה הלוגיסטית.

### 3. ROC AUC ≈ 0.489 — דומה לרגרסיה לוגיסטית
גם XGBoost לא שיפר את ה-ROC AUC. המסקנה: **הבעיה היא בנתונים עצמם**, לא בסוג המודל. עם זאת, טיפול בחוסר האיזון (`scale_pos_weight`) עשוי לשפר את F1 Macro ו-Recall קלאס 0.

### השוואה עם מודלים קודמים

| מדד | DummyClassifier | Logistic Balanced (CV) | XGBoost Baseline (CV) |
|---|---|---|---|
| Accuracy | 59.8% | 49.4% | **54.6%** |
| Recall (קלאס 0) | 0% | **47.9%** | 18.5% |
| F1 Macro | 0.37 | **0.487** | 0.460 |
| ROC AUC | 0.5 | **0.493** | 0.489 |

XGBoost baseline עדיף על Dummy ועל Logistic Baseline — אך **נופל מהלוגיסטי המאוזן (ניסוי 1)**. כוונון `scale_pos_weight` (ניסוי 3) הוא הצעד הברור הבא.

---

## ניסויים מתוכננים

| ניסוי | תיאור |
|---|---|
| ניסוי 1 ✅ | Baseline — הגדרות ברירת מחדל, ללא איזון קלאסים |
| ניסוי 2 | כוונון היפרפרמטרים: `n_estimators`, `max_depth`, `learning_rate` |
| ניסוי 3 | טיפול בחוסר איזון עם `scale_pos_weight` |
| ניסוי 4 | הסרה/הוספה של פיצ'רים מהונדסים |
| ניסוי 5 | כוונון threshold (ניתוח עקומת Precision-Recall) |
| ניסוי 6 | `GridSearchCV` / `RandomizedSearchCV` על כל הפרמטרים |

---

## קבצים

| קובץ | תיאור |
|---|---|
| `xgboost_experiment_1.py` | קוד ניסוי 1 — XGBoost baseline |
| `xgboost_cv_summary.csv` | תוצאות CV ממוצע ± סטיית תקן |
| `xgboost_experiment_1_feature_importance.csv` | חשיבות פיצ'רים מלאה |
