# XGBoost — ניבוי נטישת לקוחות

## נתונים

| פרמטר | ערך |
|---|---|
| קובץ | `E Commerce Dataset.xlsx` (גיליון: E Comm) |
| שורות | 5,630 |
| Target | `Churn` (1 = נטש, 0 = נשאר) |
| חוסר איזון | 83.2% / 16.8% |
| scale_pos_weight | 4.94 (4682 / 948) |

---

## ניסוי 1 — Baseline XGBoost + scale_pos_weight

### גישה

XGBoost בלי feature engineering ובלי threshold tuning — נקודת ייחוס נקייה מול Logistic Regression.  
חוסר האיזון מטופל דרך `scale_pos_weight=4.94` (פרמטר פנימי של XGBoost).

```python
XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=4.94,
    eval_metric="logloss",
    random_state=42,
)
```

**ולידציה:** StratifiedKFold, 5 folds, shuffle=True, random_state=42

### מאפיינים

**נומריים (13):** Tenure, WarehouseToHome, HourSpendOnApp, NumberOfDeviceRegistered,
SatisfactionScore, NumberOfAddress, Complain, OrderAmountHikeFromlastYear,
CouponUsed, OrderCount, DaySinceLastOrder, CashbackAmount, CityTier

**קטגוריים (5):** PreferredLoginDevice, PreferredPaymentMode, Gender,
PreferedOrderCat, MaritalStatus

### תוצאות (threshold=0.5)

| מדד | XGBoost Exp 1 | Logistic Exp 1 |
|---|---|---|
| Accuracy | **89.4%** | 89.4% |
| Balanced Accuracy | **88.2%** | 74.8% |
| Precision (Churn=1) | **63.5%** | 58.8% |
| Recall (Churn=1) | **86.4%** | 56.3% |
| F1 (Churn=1) | **73.2%** | 57.5% |
| F1 Macro | **83.3%** | 72.6% |
| ROC AUC | **95.2%** | 84.0% |
| PR AUC | **82.6%** | — |

### מטריצת בלבול (threshold=0.5)

|  | ניבוי: נשאר | ניבוי: נטש |
|---|---|---|
| **בפועל: נשאר** | 4,212 | 470 |
| **בפועל: נטש** | 129 | 819 |

### Threshold Sweep

| Threshold | Precision(1) | Recall(1) | F1(1) | F1 Macro |
|---|---|---|---|---|
| 0.50 | 63.5% | 86.4% | 73.2% | 83.3% |
| 0.55 | 67.2% | 83.0% | 74.3% | 84.1% |
| 0.60 | 71.7% | 80.6% | 75.9% | 85.3% |
| 0.65 | **75.1%** | 78.6% | **76.8%** | **86.0%** |
| 0.70 | 77.1% | 73.6% | 75.3% | 85.2% |

סף 0.65 נותן את ה-F1 Macro הגבוה ביותר.

### מסקנות

**הצלחות:**
- ROC AUC של 95.2% — שיפור משמעותי על Logistic Regression (84.0%)
- Recall 86.4% — המודל מזהה 86% מהלקוחות שנוטשים
- Balanced Accuracy 88.2% — הרבה יותר טוב מ-74.8% ב-Logistic Regression

**חולשות:**
- Precision רק 63.5% — על כל לקוח שנוטש נכון, יש 36.5% false positives
- 470 retained customers מסווגים שגוי כנוטשים

**השוואה ל-Logistic Regression:**
XGBoost עולה בכל המדדים ה-חשובים, בלי שום feature engineering.

---

## ניסוי 2 — Feature Engineering + Threshold Tuning

### גישה

שני שיפורים על ניסוי 1:
1. **Feature engineering** — 4 מאפיינים נגזרים (זהים ל-Logistic Regression Exp 2)
2. **Threshold tuning** — חיפוש הסף האופטימלי לפי F1 Macro דרך Precision-Recall curve

```python
# מאפיינים חדשים
AvgCashbackPerOrder  = CashbackAmount / OrderCount
IsHighComplainer     = (Complain == 1).astype(int)
LowSatisfaction      = (SatisfactionScore <= 2).astype(int)
DaysSinceOrderBucket = "recent" / "medium" / "long" / "unknown"
```

**מאפיינים סה"כ:** 16 נומריים, 6 קטגוריים (במקום 13+5 בניסוי 1)

### תוצאות (threshold=0.5)

| מדד | XGBoost Exp 2 | XGBoost Exp 1 |
|---|---|---|
| Accuracy | 89.5% | 89.4% |
| Balanced Accuracy | 88.3% | 88.2% |
| Precision (Churn=1) | 63.9% | 63.5% |
| Recall (Churn=1) | 86.5% | 86.4% |
| F1 (Churn=1) | 73.5% | 73.2% |
| F1 Macro | 83.5% | 83.3% |
| ROC AUC | **95.2%** | 95.2% |
| PR AUC | **82.6%** | 82.6% |

### תוצאות (threshold=0.66 — אופטימלי לפי F1 Macro)

| מדד | threshold=0.5 | threshold=0.66 |
|---|---|---|
| Accuracy | 89.5% | **92.3%** |
| Balanced Accuracy | 88.3% | 86.6% |
| Precision (Churn=1) | 63.9% | **76.7%** |
| Recall (Churn=1) | 86.5% | 78.1% |
| F1 (Churn=1) | 73.5% | **77.4%** |
| F1 Macro | 83.5% | **86.4%** |
| ROC AUC | 95.2% | 95.2% |

### מטריצת בלבול (threshold=0.66)

|  | ניבוי: נשאר | ניבוי: נטש |
|---|---|---|
| **בפועל: נשאר** | 4,457 | 225 |
| **בפועל: נטש** | 208 | 740 |

### מסקנות

**הצלחות:**
- Threshold tuning שיפר F1 Macro מ-83.5% ל-86.4%
- Precision קפצה מ-63.9% ל-76.7% — false positives ירדו מ-463 ל-225
- Accuracy עלתה ל-92.3%

**Feature Engineering:**
- שיפור שולי ביחס לניסוי 1 (ROC AUC זהה) — XGBoost לומד בעצמו מרבית האינטראקציות
- הערך האמיתי של threshold tuning הוא גדול יותר מ-feature engineering בהקשר זה

**מה לא השתנה:** ROC AUC נשאר 95.2% — הסיווג הגולמי זהה; רק נקודת ההפרדה שונתה

---

## השוואה כללית בין המודלים

| מודל | ROC AUC | F1 Macro | Recall(1) | Precision(1) |
|---|---|---|---|---|
| Logistic Reg. Exp 1 (baseline) | 84.0% | 72.6% | 56.3% | 58.8% |
| Logistic Reg. Exp 2 (thresh=0.5) | ~84% | ~74% | ~80% | ~55% |
| XGBoost Exp 1 (thresh=0.5) | 95.2% | 83.3% | 86.4% | 63.5% |
| **XGBoost Exp 2 (thresh=0.66)** | **95.2%** | **86.4%** | 78.1% | **76.7%** |

---

## קבצים

| קובץ | תיאור |
|---|---|
| `xgboost_experiment_1.py` | ניסוי 1 — Baseline + scale_pos_weight |
| `xgboost_experiment_1_summary.csv` | תוצאות ניסוי 1 |
| `xgboost_experiment_2.py` | ניסוי 2 — Feature Engineering + Threshold Tuning |
| `xgboost_experiment_2_summary.csv` | תוצאות ניסוי 2 |
