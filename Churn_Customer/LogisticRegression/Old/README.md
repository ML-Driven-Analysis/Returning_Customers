# Logistic Regression — E-Commerce Customer Churn

## מטרה

ניסוי בסיסי ראשון על **נתונים אמיתיים**: האם Logistic Regression יכול לנבא נטישת לקוח בחנות אונליין?

זהו הניסוי הראשון בפרויקט שנוצר על נתונים אמיתיים (לאחר שכל הניסויים על הנתונים הסינתטיים הגיעו ל-ROC AUC ≈ 0.5).

---

## נתונים

| פרמטר | ערך |
|---|---|
| קובץ | `../dataset/E Commerce Dataset.xlsx` |
| שורות | 5,630 |
| Target | `Churn` (1=נטש, 0=נשאר) |
| חוסר איזון | 83.2% לא נטשו (4,682) / 16.8% נטשו (948) — יחס 4.94:1 |

---

## ניסוי 1 — Baseline Logistic Regression

### גישה

Pipeline בסיסי ללא feature engineering וללא טיפול בחוסר איזון:

```
ColumnTransformer:
  numeric (13)   → SimpleImputer(median) + StandardScaler
  categorical (5) → SimpleImputer(most_frequent) + OneHotEncoder
→ LogisticRegression(max_iter=1000, solver='lbfgs')

StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### פיצ'רים

**Numeric (13)**: Tenure, WarehouseToHome, HourSpendOnApp, NumberOfDeviceRegistered, SatisfactionScore, NumberOfAddress, Complain, OrderAmountHikeFromlastYear, CouponUsed, OrderCount, DaySinceLastOrder, CashbackAmount, CityTier

**Categorical (5)**: PreferredLoginDevice, PreferredPaymentMode, Gender, PreferedOrderCat, MaritalStatus

### תוצאות (5-Fold Stratified CV)

| מדד | ממוצע | סטיית תקן | הערה |
|---|---|---|---|
| Accuracy | 89.4% | ±0.29% | מטעה — מושפע מחוסר איזון |
| Balanced Accuracy | **74.8%** | ±0.66% | דיוק אמיתי |
| Precision (נוטשים) | 76.9% | ±1.66% | כשצופה נטישה — צודק 77% |
| Recall (נוטשים) | **52.8%** | ±1.41% | מפספס 47% מהנוטשים ⚠️ |
| F1 (נוטשים) | 62.5% | ±1.06% | ממוצע Precision+Recall |
| **F1 Macro** | **78.2%** | ±0.61% | ממוצע בין שתי המחלקות |
| **ROC AUC** | **89.0%** | ±0.36% | המדד הכולל החשוב ביותר |

### מטריצת בלבול (OOF מצטברת — 5,630 רשומות)

|  | חזוי: נשאר (0) | חזוי: נטש (1) |
|---|---|---|
| **אמיתי: נשאר (0)** | 4,531 ✓ | 151 ✗ |
| **אמיתי: נטש (1)** | **448 ✗** | 500 ✓ |

> 448 נוטשים **לא זוהו** (False Negatives) — הבעיה המרכזית שניסוי 2 יפתור.

---

## מסקנות ניסוי 1

### ✅ הצלחה — Signal אמיתי קיים

**ROC AUC = 0.890** לעומת 0.50 בנתונים הסינתטיים — הוכחה שהדאטאסט הזה מכיל **קשר סטטיסטי אמיתי** בין הפיצ'רים ל-label. המודל הבסיסי ביותר, ללא כל כוונון, כבר מגיע לביצועים גבוהים.

### ⚠️ חולשה — Recall נמוך על נוטשים

**Recall(Churn=1) = 52.8%** — המודל מפספס **448 מתוך 948 נוטשים** (47%). זה נובע ישירות מחוסר האיזון: המודל "בוחר" לנבא "לא נטש" כברירת מחדל כי זה נכון 83% מהזמן.

בעולם העסקי זוהי בעיה קריטית: **כל נוטש שלא זוהה = לקוח שאבד ללא התראה**.

### 📊 הדיוק מטעה

Accuracy = 89.4% נראה מרשים, אך זה בגלל שהמודל פשוט מנבא "לא נטש" ברוב המקרים. **Balanced Accuracy = 74.8%** היא המדד האמיתי.

---

## ניסוי 2 — class_weight + Feature Engineering + Threshold Tuning

### גישה

```
Feature Engineering: +4 פיצ'רים חדשים (16 numeric, 6 categorical)
GridSearchCV: C in [0.01, 0.1, 1, 10, 100] — best C=10
LogisticRegression(class_weight='balanced', C=10)
Threshold tuning: optimal = 0.7323 (ממקסם F1 Macro)
```

### פיצ'רים חדשים שנוספו

| פיצ'ר | חישוב | משמעות |
|---|---|---|
| `AvgCashbackPerOrder` | `CashbackAmount / OrderCount` | קאשבק ממוצע לפי הזמנה |
| `IsHighComplainer` | `Complain == 1` | הגיש תלונה |
| `LowSatisfaction` | `SatisfactionScore <= 2` | שביעות רצון נמוכה מאוד |
| `DaysSinceOrderBucket` | recent/medium/long | גיל ההזמנה האחרונה |

### GridSearchCV — בחירת C (3-Fold, scoring=ROC AUC)

| C | ROC AUC | סטיית תקן |
|---|---|---|
| 0.01 | 88.51% | ±0.20% |
| 0.1 | 88.91% | ±0.34% |
| 1 | 89.07% | ±0.38% |
| **10** | **89.08%** | ±0.40% |
| 100 | 89.07% | ±0.41% |

> C=10 נבחר — פחות רגולריזציה משפרת מעט, מעבר ל-100 לא מוסיף דבר.

### תוצאות — Threshold = 0.5 (ברירת מחדל)

| מדד | ערך | לעומת ניסוי 1 |
|---|---|---|
| Accuracy | 80.6% | -8.8% |
| Precision (נוטשים) | 45.8% | -31.1% |
| **Recall (נוטשים)** | **81.8%** | **+29.0% ✅** |
| F1 (נוטשים) | 58.7% | -3.8% |
| F1 Macro | 73.0% | -5.2% |
| ROC AUC | **89.0%** | ~0 |

### תוצאות — Threshold = 0.73 (אופטימלי)

| מדד | ערך | לעומת ניסוי 1 |
|---|---|---|
| Accuracy | 87.9% | -1.5% |
| Precision (נוטשים) | 63.9% | -13.0% |
| **Recall (נוטשים)** | **64.9%** | **+12.1% ✅** |
| F1 (נוטשים) | 64.4% | +1.9% |
| **F1 Macro** | **78.5%** | **+0.3% ✅** |
| ROC AUC | **89.0%** | ~0 |

### מטריצת בלבול — Threshold אופטימלי (0.73)

|  | חזוי: נשאר (0) | חזוי: נטש (1) |
|---|---|---|
| **אמיתי: נשאר (0)** | 4,334 ✓ | 348 ✗ |
| **אמיתי: נטש (1)** | **333 ✗** | 615 ✓ |

> ירידה מ-448 נוטשים שפוספסו (ניסוי 1) ל-333 — **שיפור של 115 לקוחות נוספים שזוהו**.

---

## השוואה סופית

| מדד | ניסוי 1 (baseline) | ניסוי 2 (threshold=0.5) | ניסוי 2 (threshold=0.73) |
|---|---|---|---|
| ROC AUC | **89.0%** | 89.0% | 89.0% |
| Recall (נוטשים) | 52.8% | **81.8%** | 64.9% |
| Precision (נוטשים) | **76.9%** | 45.8% | 63.9% |
| F1 Macro | 78.2% | 73.0% | **78.5%** |
| נוטשים שפוספסו | 448 | **173** | 333 |

---

## מסקנות

### ROC AUC — לא השתנה (89%)
רמת ה-AUC זהה בשני הניסויים. המשמעות: **הכוח הסטטיסטי של המודל לא השתפר** — ה-feature engineering לא הוסיף signal חדש. ה-AUC הוא מדד שעמיד לסף ולמשקלות, ולכן הוא מייצג את "תקרת" הביצועים של הגישה הזו.

### Threshold=0.5 עם class_weight — הכי הרבה נוטשים נתפסים
Recall קפץ מ-52.8% ל-**81.8%** — נתפסו **775 מתוך 948 נוטשים**. המחיר: Precision ירד ל-45.8% — כמעט מחצית מהתראות ה"נטישה" הן שגויות. **נכון לבחור אם עלות פנייה לא-נוטש זולה מעלות אובדן נוטש**.

### Threshold=0.73 — האיזון הטוב ביותר
F1 Macro מקסימלי (78.5%), Recall ו-Precision כמעט שווים (65%/64%). **זו ההגדרה המאוזנת** לשימוש כללי.

### הבחירה תלויה בהחלטה עסקית
| מצב | Threshold מומלץ |
|---|---|
| קמפיין שימור — כל פנייה עולה כסף | 0.73 (מאוזן) |
| פנייה זולה (אימייל/פוש) — עדיף לא לפספס | 0.5 (Recall גבוה) |

---

## קבצים

| קובץ | תיאור |
|---|---|
| `logistic_experiment_1.py` | ניסוי 1 — baseline |
| `logistic_experiment_1_summary.csv` | תוצאות ניסוי 1 |
| `logistic_experiment_2.py` | ניסוי 2 — balanced + tuning |
| `logistic_experiment_2_summary.csv` | תוצאות ניסוי 2 |
