# עיצוב: ניסויי שיפור מודל רגרסיה לוגיסטית

**תאריך:** 2026-06-01  
**פרויקט:** Returning Customers — LogisticRegression  
**בעיה:** ROC AUC = 0.475, המודל מנבא כמעט תמיד קלאס 1

---

## רקע

המודל הבסיסי (`logistic_regression.py`) מציג ביצועים גרועים מניחוש אקראי. שורש הבעיה הוא שילוב של חוסר איזון בין קלאסים (60/40) ללא טיפול, ופיצ'רים מקורלים שמוסיפים רעש. שני ניסויים מבודדים יבחנו את השפעת כל שינוי בנפרד.

---

## מבנה קבצים

```
LogisticRegression/
├── logistic_regression.py          ← Baseline (לא נוגעים)
├── logistic_experiment_1.py        ← Exp1: class_weight='balanced'
└── logistic_experiment_2.py        ← Exp2: Exp1 + הסרת פיצ'רים מקורלים
```

כל קובץ שומר מודל נפרד:
- `logistic_regression_returning_customer_model.pkl` (קיים)
- `logistic_experiment_1_model.pkl`
- `logistic_experiment_2_model.pkl`

---

## ניסוי 1: טיפול בחוסר איזון קלאסים

**שינוי יחיד:**
```python
LogisticRegression(max_iter=3000, random_state=42, class_weight='balanced')
```

**מטרה:** לאלץ את המודל לשים משקל שווה על שני הקלאסים, כך שלא יטה לנבא תמיד קלאס 1.

**תוצאה מצופה:** עלייה ב-Recall של קלאס 0 (מ-3%), ירידה ב-Recall של קלאס 1, שיפור ב-ROC AUC מעבר ל-0.5.

---

## ניסוי 2: ניסוי 1 + ניקוי פיצ'רים מקורלים

**מצטבר על גבי ניסוי 1.** פיצ'רים שמוסרים:

| פיצ'ר | סיבה |
|---|---|
| `Original_Amount` | = `Unit_Price × Quantity` — מקורל גבוה עם `Total_Amount` |
| `Long_Session` | גרסה בינארית של `Session_Duration_Minutes` (כבר קיים) |
| `Many_Pages_Viewed` | גרסה בינארית של `Pages_Viewed` (כבר קיים) |
| `Fast_Delivery` | גרסה בינארית של `Delivery_Time_Days` (כבר קיים) |
| `High_Rating` | גרסה בינארית של `Customer_Rating` (כבר קיים) |

**פיצ'רים שנשארים** (מוסיפים מידע חדש):
- `Has_Discount`, `Discount_Rate`, `Amount_Per_Item`, `Pages_Per_Minute`, `IsWeekend`

**מימוש:** הוספת שמות הפיצ'רים לרשימת `columns_to_drop` שבסעיף 5 של הסקריפט (לאחר קריאה ל-`extract_features()`), כך שהם לא ייכנסו ל-`X`.

**מטרה:** להפחית רעש ומולטיקולינאריות — לבדוק האם המודל לומד כלל אחרי שהפיצ'רים נקיים יותר.

---

## תיעוד ב-README

מוסיפים לסוף `LogisticRegression/README.md` קטע **"ניסויים ושיפורים"** עם:

1. טבלת השוואה מרכזית (Baseline / Exp1 / Exp2) — המדדים: Accuracy, Recall (0), Recall (1), F1 Macro, ROC AUC
2. תת-סעיף לכל ניסוי: שינויים שבוצעו, מטריצת בלבול, מסקנה קצרה

---

## קריטריוני הצלחה

- ROC AUC > 0.5 לאחר ניסוי 1
- Recall של קלאס 0 > 30% לאחר ניסוי 1
- F1 Macro > 0.45 לאחר ניסוי 2
