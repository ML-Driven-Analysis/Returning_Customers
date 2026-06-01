# עיצוב: מודל Baseline — DummyClassifier

**תאריך:** 2026-06-01
**פרויקט:** Returning Customers — DummyClassifier
**מטרה:** יצירת מודל baseline טיפש לצורך השוואה עם המודל הלוגיסטי

---

## רקע

מודל הרגרסיה הלוגיסטית השיג ROC AUC = 0.475 — גרוע מניחוש אקראי. כדי להבין אם הוא בכלל לומד משהו, צריך להשוות אותו מול `DummyClassifier` שמנבא תמיד את הקלאס הנפוץ (1 = לקוח חוזר, 59.8% מהנתונים).

---

## מבנה קבצים

```
DummyClassifier/
├── dummy_classifier.py          ← סקריפט הסיווג הטיפש
├── dummy_classifier_model.pkl   ← המודל השמור (joblib)
└── README.md                    ← תיאור, תוצאות, מסקנות בעברית
```

---

## הסקריפט

**אותה חלוקת נתונים** כמו ה-baseline הלוגיסטי:
- `DATA_FILE`: `dataset/ecommerce_customer_behavior_dataset.csv`
- `train_test_split(test_size=0.2, random_state=42, stratify=y)`
- תווית: `Is_Returning_Customer`

**אין** feature engineering, preprocessing, או scaling — DummyClassifier אינו משתמש בפיצ'רים.

**מודל:**
```python
DummyClassifier(strategy="most_frequent", random_state=42)
```

**הערכה זהה ללוגיסטי:**
- Accuracy, Precision, Recall, F1 Score (zero_division=0)
- ROC AUC
- Confusion Matrix
- Classification Report

**שמירת מודל:** `DummyClassifier/dummy_classifier_model.pkl`

---

## תוצאות צפויות (לפני הרצה)

| מדד | ערך צפוי |
|---|---|
| Accuracy | ~59.8% (שיעור הקלאס הנפוץ) |
| Recall (קלאס 0) | 0% (לעולם לא מנבא 0) |
| Recall (קלאס 1) | 100% (תמיד מנבא 1) |
| ROC AUC | ~0.5 (ניחוש אקראי) |

---

## README

יכלול (בעברית):
1. תיאור המודל ומטרתו כ-baseline
2. טבלת תוצאות אמיתיות (לאחר הרצה)
3. מסקנות: מה המספרים אומרים ומה ניתן ללמוד מהשוואה עם הלוגיסטי

---

## עדכון LogisticRegression/README.md

לאחר הרצת ה-DummyClassifier, יתווסף ל-`LogisticRegression/README.md` סעיף **"השוואה מול Baseline טיפש"** עם:
- טבלת ROC AUC / F1 Macro / Recall קלאס 0 בין שלושת המודלים (Dummy, Baseline LR, Exp1, Exp2)
- מסקנה: האם הלוגיסטי עוקף את ה-Dummy? בכמה?

---

## קריטריוני הצלחה

- הסקריפט רץ ומייצר מדדים
- README מלא עם תוצאות אמיתיות ומסקנות
- LogisticRegression/README.md מעודכן עם השוואה
