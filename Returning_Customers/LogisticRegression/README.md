# מודל רגרסיה לוגיסטית – חיזוי לקוחות חוזרים

## תיאור כללי

מודל **רגרסיה לוגיסטית (Logistic Regression)** שנועד לחזות האם לקוח ב-e-commerce יהיה **לקוח חוזר** (Returning Customer) על בסיס נתוני הזמנה, התנהגות גלישה, מאפיינים דמוגרפיים ופרטי משלוח.

---

## מקור הנתונים

| פרמטר | ערך |
|---|---|
| קובץ | `dataset/ecommerce_customer_behavior_dataset.csv` |
| מספר רשומות | 5,000 |
| מספר עמודות מקוריות | 18 |
| ערכים חסרים | אין |

---

## עמודות המקור

| עמודה | תיאור |
|---|---|
| `Order_ID`, `Customer_ID` | מזהים (הוסרו מהאימון) |
| `Date` | תאריך ההזמנה |
| `Age` | גיל הלקוח |
| `Gender` | מגדר |
| `City` | עיר |
| `Product_Category` | קטגוריית מוצר |
| `Unit_Price`, `Quantity`, `Discount_Amount`, `Total_Amount` | פרטי עסקה |
| `Payment_Method` | אמצעי תשלום |
| `Device_Type` | סוג מכשיר (Mobile / Desktop) |
| `Session_Duration_Minutes`, `Pages_Viewed` | התנהגות גלישה |
| `Delivery_Time_Days` | ימי משלוח |
| `Customer_Rating` | דירוג לקוח (1–5) |
| `Is_Returning_Customer` | **תווית היעד** |

---

## הנדסת פיצ'רים (Feature Engineering)

מעבר לעמודות המקור, נוספו הפיצ'רים הבאים:

### פיצ'רי תאריך
- `Year`, `Month`, `DayOfWeek`, `DayOfMonth`, `Quarter` – פירוק התאריך לרכיבים
- `IsWeekend` – האם ההזמנה בסוף שבוע (0/1)

### פיצ'רי סכום
- `Original_Amount` = `Unit_Price × Quantity` – סכום לפני הנחה
- `Has_Discount` – האם קיימת הנחה (0/1)
- `Discount_Rate` – שיעור ההנחה מהסכום המקורי
- `Amount_Per_Item` – עלות ממוצעת לפריט

### פיצ'רי גלישה
- `Pages_Per_Minute` – קצב גלישה
- `Long_Session` – האם הסשן ארוך מהחציון (0/1)
- `Many_Pages_Viewed` – האם מספר העמודים גדול מהחציון (0/1)

### פיצ'רי משלוח ושביעות רצון
- `Fast_Delivery` – האם המשלוח מהיר מהחציון (0/1)
- `High_Rating` – האם הדירוג ≥ 4 (0/1)

**סה"כ פיצ'רים לפני One-Hot Encoding: 29**

---

## ארכיטקטורת הפייפליין

```
Raw Data
    ↓
extract_features()
    ↓
ColumnTransformer
 ├── Numeric: SimpleImputer(median) → StandardScaler
 └── Categorical: SimpleImputer(most_frequent) → OneHotEncoder
    ↓
LogisticRegression(max_iter=3000, random_state=42)
```

---

## חלוקת נתונים

| קבוצה | גודל | אחוז |
|---|---|---|
| Train | 4,000 | 80% |
| Test | 1,000 | 20% |

חלוקה **מאוזנת (stratify)** לפי התווית.

---

## התפלגות תווית היעד

| קלאס | מספר | אחוז |
|---|---|---|
| 1 (לקוח חוזר) | 2,990 | 59.8% |
| 0 (לא חוזר) | 2,010 | 40.2% |

קיים **חוסר איזון מתון** לטובת לקוחות חוזרים.

---

## תוצאות הערכה

| מדד | ערך |
|---|---|
| **Accuracy** | 58.5% |
| **Precision** (class 1) | 59.5% |
| **Recall** (class 1) | 95.8% |
| **F1 Score** (class 1) | 73.4% |
| **ROC AUC** | **0.475** ⚠️ |

### מטריצת בלבול

|  | חזוי 0 | חזוי 1 |
|---|---|---|
| **אמיתי 0** | 12 | 390 |
| **אמיתי 1** | 25 | 573 |

---

## הפיצ'רים המשפיעים ביותר (Top 15)

| פיצ'ר | מקדם |
|---|---|
| `City_Bursa` | +0.270 |
| `Month` | −0.151 |
| `Device_Type_Mobile` | +0.137 |
| `Quarter` | +0.135 |
| `City_Ankara` | −0.130 |
| `Payment_Method_Digital Wallet` | +0.129 |
| `Gender_Other` | +0.114 |
| `Product_Category_Food` | +0.110 |
| `Product_Category_Toys` | +0.104 |
| `Delivery_Time_Days` | −0.102 |
| `Unit_Price` | +0.092 |
| `Payment_Method_Debit Card` | +0.090 |
| `City_Antalya` | +0.071 |
| `Fast_Delivery` | −0.070 |
| `Many_Pages_Viewed` | +0.068 |

---

## ניתוח הפלט – מסקנות עיקריות

### 1. המודל כמעט תמיד מנבא "לקוח חוזר"
מתוך 402 לקוחות שאינם חוזרים בקבוצת הבדיקה, המודל זיהה נכון רק **12** (Recall = 3%). המודל מנבא כמעט תמיד קלאס 1, ולכן אין לו ערך מעשי לזיהוי לקוחות לא-חוזרים.

### 2. ROC AUC מתחת ל-0.5 – גרוע מניחוש אקראי
ערך ROC AUC של **0.475** פירושו שהמודל **גרוע מהטלת מטבע**. זהו אינדיקטור חמור לכך שיש בעיה מבנית – קורלציה הפוכה בין הפיצ'רים לתווית, או שהנתונים אינם מספיקים כדי לאפשר הפרדה לינארית.

### 3. Accuracy מטעה
Accuracy של 58.5% נראית סבירה, אך היא משקפת רק את העובדה שרוב הנתונים הם קלאס 1 (59.8%). מסווג טריוויאלי שמנבא תמיד "1" ישיג אחוז דומה.

### 4. הפיצ'רים הדומיננטיים הם קטגוריאליים גיאוגרפיים
עיר (Bursa, Ankara) ואמצעי תשלום (Digital Wallet) מובילים את הרשימה – אך מקדמים קטנים (<0.3) מצביעים על כך שאין פיצ'ר חזק באמת.

---


## קבצים

| קובץ | תיאור |
|---|---|
| `logistic_regression.py` | קוד המודל המלא (baseline) |
| `logistic_regression_returning_customer_model.pkl` | מודל baseline (joblib) |
| `logistic_experiment_1.py` | ניסוי 1: class_weight='balanced' |
| `logistic_experiment_1_model.pkl` | מודל ניסוי 1 (joblib) |
| `logistic_experiment_2.py` | ניסוי 2: balanced + הסרת פיצ'רים מקורלים |
| `logistic_experiment_2_model.pkl` | מודל ניסוי 2 (joblib) |
| `logistic_experiment_3.py` | ניסוי 3: Cross-Validation 5-Fold על שלושת ההגדרות |
| `logistic_cross_validation_summary.csv` | טבלת תוצאות CV מסוכמת |
| `best_logistic_coefficients.csv` | מקדמי המודל הטוב ביותר לפי CV |
| `logistic1.png` | גרף/תמונה נלווית |

---

## ניסויים ושיפורים

### טבלת השוואה מרכזית

| מדד | Baseline | ניסוי 1 | ניסוי 2 |
|---|---|---|---|
| Accuracy | 58.5% | 49.0% | 47.8% |
| Recall (קלאס 0) | 3% | 51% | 52% |
| Recall (קלאס 1) | 95.8% | 47% | 45% |
| F1 Macro | 0.39 | 0.49 | 0.48 |
| ROC AUC | 0.475 | 0.475 | 0.478 |

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
| **אמיתי 0** | 207 | 195 |
| **אמיתי 1** | 315 | 283 |

**מסקנה:** הוספת `class_weight='balanced'` שינתה דרמטית את ההתנהגות — Recall קלאס 0 קפץ מ-3% ל-51%. המודל כעת מאוזן בין הקלאסים. ROC AUC נשאר 0.475 מכיוון ש-ROC AUC הוא מדד **בלתי-תלוי-ספף** המשקף את הכוח ההפרדתי הגולמי של הפיצ'רים — כוח שאיזון המשקלות אינו יכול לשפר כאשר הפיצ'רים עצמם חלשים.

---

### ניסוי 2: class_weight='balanced' + הסרת פיצ'רים מקורלים

**קובץ:** `logistic_experiment_2.py`

**שינויים שבוצעו:**
- `class_weight='balanced'` (מניסוי 1)
- הוסרו: `Original_Amount`, `Long_Session`, `Many_Pages_Viewed`, `Fast_Delivery`, `High_Rating`
- מספר פיצ'רים: 29 → 24

**מטריצת בלבול:**

|  | חזוי 0 | חזוי 1 |
|---|---|---|
| **אמיתי 0** | 208 | 194 |
| **אמיתי 1** | 328 | 270 |

**מסקנה:** הסרת הפיצ'רים המקורלים שיפרה מעט את ROC AUC (0.475 → 0.478) ואת Recall קלאס 0 (51% → 52%), אך השיפור שולי. המסקנה המרכזית: הנתונים עצמם אינם מאפשרים הפרדה לינארית טובה — הפיצ'רים הקיימים אינם מנבאים חזק מספיק. השלב הבא המומלץ: מודלים לא-לינאריים (Random Forest / XGBoost).

---

## השוואה מול Baseline טיפש (DummyClassifier)

| מדד | DummyClassifier | Logistic Baseline | ניסוי 1 | ניסוי 2 |
|---|---|---|---|---|
| Accuracy | 59.8% | 58.5% | 49.0% | 47.8% |
| Recall (קלאס 0) | 0% | 3% | 51% | 52% |
| F1 Macro | 0.37 | 0.39 | 0.49 | 0.48 |
| ROC AUC | 0.5 | 0.475 ⚠️ | 0.475 | 0.478 |

### תובנות

**1. הלוגיסטי ה-baseline כמעט זהה ל-DummyClassifier**
הרגרסיה הלוגיסטית ללא איזון קלאסים (Baseline) השיגה Accuracy של 58.5% לעומת 59.8% של ה-Dummy, ו-ROC AUC של 0.475 — נמוך מ-0.5 של ה-Dummy. המסקנה: ה-baseline ה"לא מאוזן" לא לומד דבר מעבר להטיה לקלאס 1, ובמדדים מסוימים **גרוע מהמודל הטיפש**.

**2. class_weight='balanced' הוא המינימום הנדרש**
רק לאחר הוספת `class_weight='balanced'` (ניסוי 1) המודל מתחיל לעקוף את ה-Dummy בצורה ברורה: Recall קלאס 0 קפץ מ-0% ל-51%, F1 Macro עלה מ-0.37 ל-0.49. **כל מודל שמוגש בפרויקט הזה חייב לעקוף לפחות את ניסוי 1.**

**3. ROC AUC ≈ 0.475–0.478 — הפיצ'רים חלשים**
גם לאחר כל השיפורים, ROC AUC נשאר קרוב ל-0.5. ה-DummyClassifier מאשר שהבעיה אינה בטיפול בחוסר האיזון אלא בעוצמת הפיצ'רים עצמם. הצעד הבא המומלץ: מודלים לא-לינאריים (Random Forest / XGBoost) שיכולים ללכוד קשרים מורכבים יותר.
