# CustomerRFM — Customer-Level RFM Behavioral Profiling

## מטרה

כל הניסויים הקודמים (Logistic Regression, XGBoost ×5) השיגו **ROC AUC ≈ 0.5** — כמעט אקראי לחלוטין. ההשערה: המודלים נכשלו כי הם מנסים לנבא נאמנות לקוח מ**עסקה בודדת**, בעוד נאמנות מוגדרת על ידי **דפוסים לאורך זמן**.

ניסוי זה בדק את ההשערה: האם שימוש בהיסטוריית רכישות מלאה לכל לקוח (פרופיל RFM) ישפר את הביצועים?

---

## הגילוי שאפשר את הניסוי

הקובץ `for_test.csv` בתיקייה הראשית מכיל גרסת **multi-order** של אותם 5,000 לקוחות:

| פרמטר | ערך |
|---|---|
| סה"כ עסקאות | 17,049 |
| לקוחות ייחודיים | 5,000 |
| ממוצע עסקאות ללקוח | 3.41 |
| טווח עסקאות ללקוח | 1–10 |

---

## גישת המודל

### מקורות נתונים

| קובץ | שימוש |
|---|---|
| `for_test.csv` | Feature engineering — aggregation של כל עסקאות הלקוח |
| `dataset/ecommerce_customer_behavior_dataset.csv` | Label בלבד — `Is_Returning_Customer` מקורי לכל לקוח |

**תוצאת ה-join**: 5,000 שורות × 29 פיצ'רים ברמת לקוח.

### פיצ'רים (29)

| קטגוריה | פיצ'רים |
|---|---|
| **Frequency** | `n_orders`, `order_span_days` |
| **Recency** | `avg_gap_days`, `min_gap_days`, `max_gap_days` |
| **Monetary** | `total_spend`, `avg_order_value`, `max_order_value`, `std_order_value`, `total_discount`, `avg_discount`, `discount_usage_rate` |
| **Product** | `n_unique_categories`, `favorite_category`, `avg_unit_price`, `avg_quantity` |
| **Session** | `avg_session_duration`, `std_session_duration`, `avg_pages_viewed` |
| **Payment** | `favorite_payment`, `favorite_device` |
| **Satisfaction** | `avg_rating`, `min_rating`, `avg_delivery_days` |
| **Demographics** | `Age`, `Gender`, `City` |
| **Time** | `first_order_month`, `most_common_dow` |

### מודל

**LightGBM** — נבחר בגלל:
- טיפול native בפיצ'רים קטגוריאליים (ללא OneHotEncoder)
- `class_weight='balanced'` מובנה
- מהיר ויעיל על tabular data

```python
LGBMClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    num_leaves=15, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=20, class_weight="balanced", random_state=42
)
```

### ולידציה

StratifiedKFold (n_splits=5, shuffle=True, random_state=42)

---

## תוצאות

### ממוצעי 5 קפלים

| מדד | ממוצע | סטיית תקן |
|---|---|---|
| Accuracy | 51.3% | ±1.58% |
| Balanced Accuracy | 50.0% | ±1.61% |
| Precision (קלאס 0) | 40.2% | ±1.70% |
| Recall (קלאס 0) | 43.4% | ±3.30% |
| F1 (קלאס 0) | 0.417 | ±0.023 |
| Precision (קלאס 1) | 59.8% | ±1.43% |
| Recall (קלאס 1) | 56.7% | ±2.72% |
| F1 (קלאס 1) | 0.582 | ±0.018 |
| **F1 Macro** | **0.499** | ±0.016 |
| **ROC AUC** | **0.502** | ±0.022 |

### מטריצת בלבול (OOF מצטברת — 5,000 רשומות)

|  | חזוי 0 | חזוי 1 |
|---|---|---|
| **אמיתי 0** | 872 | 1,138 |
| **אמיתי 1** | 1,296 | 1,694 |

### Top 10 Feature Importances

| פיצ'ר | חשיבות |
|---|---|
| `Age` | 203 |
| `avg_unit_price` | 192 |
| `avg_pages_viewed` | 192 |
| `avg_session_duration` | 169 |
| `max_order_value` | 167 |
| `std_session_duration` | 157 |
| `avg_delivery_days` | 157 |
| `total_spend` | 154 |
| `order_span_days` | 148 |
| `avg_quantity` | 142 |

---

## מסקנה סופית — הוכחה חד-משמעית

### השוואה כוללת — כל הניסויים בכל הפרויקט

| מודל | גישה | F1 Macro | ROC AUC |
|---|---|---|---|
| DummyClassifier | Baseline | 0.370 | 0.500 |
| Logistic Regression Balanced | Transaction-level | 0.487 | 0.493 |
| XGBoost Baseline | Transaction-level | 0.460 | 0.489 |
| XGBoost + scale_pos_weight | Transaction-level | 0.487 | 0.485 |
| XGBoost + RandomizedSearchCV | Transaction-level | 0.496 | 0.498 |
| XGBoost + Threshold Tuning | Transaction-level | 0.489 | 0.487 |
| XGBoost + Minimal Features | Transaction-level | 0.501 | 0.502 |
| **LightGBM + RFM (ניסוי זה)** | **Customer-level** | **0.499** | **0.502** |

### המסקנה

**ה-label `Is_Returning_Customer` בנתונים אלו אינו קשור לאף פיצ'ר — לא ברמת עסקה ולא ברמת לקוח.**

ניסוי ה-RFM הוא הניסוי הקונקלוסיבי ביותר: הוא השתמש ב-29 פיצ'רים מצטברים מ-17,049 עסקאות, כולל תדירות, מוניטרי, recency, התנהגות גלישה, שביעות רצון — וה-ROC AUC נותר 0.502.

**המסקנה היחידה האפשרית**: ה-label הוגרל רנדומלית בנתונים הסינתטיים ואין לו קשר סטטיסטי לאף אחד מהפיצ'רים הקיימים. זהו מגבלת הנתונים, לא של שיטת המידול.

### הצעד הבא

כדי לפתור בעיית חיזוי אמיתית של לקוח חוזר, יש צורך ב:
- **נתונים אמיתיים** (לא סינתטיים) עם קשר אמיתי בין פיצ'רים ל-label
- **פיצ'רים נוספים** כגון: היסטוריית מייל/פרסום, חברות בתכנית loyalty, ערוץ גיוס, זמן מאז הרכישה האחרונה

---

## קבצים

| קובץ | תיאור |
|---|---|
| `rfm_experiment_1.py` | קוד מלא — RFM feature engineering + LightGBM |
| `rfm_experiment_1_summary.csv` | תוצאות CV ממוצע ± סטיית תקן |
| `rfm_experiment_1_feature_importance.csv` | חשיבות 29 פיצ'רים |
