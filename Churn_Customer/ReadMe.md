<div dir="rtl" align="right">

# Churn Customer – Project Structure

מסמך זה מתאר את מבנה תיקיית `Churn_Customer` ואת מטרת כל תיקייה, סקריפט וקובץ תוצאה בפרויקט חיזוי נטישת לקוחות.

---

## מבנה כללי

```text
Churn_Customer
│
├── evaluation_utils.py
├── catboost
├── dataset
├── dummy_model
├── LogisticRegression
├── RandomForest
└── XGBoost
```

תיקיית `Churn_Customer` מרכזת את כלל הניסויים שבוצעו על דאטהסט נטישת הלקוחות, החל ממודל בסיסי פשוט ועד מודלים מתקדמים יותר כגון `Random Forest`, `XGBoost` ו־`CatBoost`.

---

# evaluation_utils.py

קובץ עזר משותף לכל המודלים בפרויקט.

מטרתו לרכז את פונקציות ההערכה במקום אחד, כדי שכל המודלים ייבחנו באותו פורמט ובאותם מדדים.

הקובץ כולל פונקציות עבור:

<div dir="rtl" align="right">

• חישוב מדדי ביצועים עבור תחזיות קיימות.
• הערכת מודלים באמצעות Cross Validation.
• חישוב מדדים עבור כמה ערכי Threshold שונים.
• הדפסת תוצאות בצורה אחידה וברורה.
• הפקת מדדים כגון Accuracy, Balanced Accuracy, Precision, Recall, F1, F1 Macro, ROC AUC, PR AUC ו־Confusion Matrix.

</div>

---

# תיקיית dataset

```text
dataset
│
├── E Commerce Dataset.xlsx
├── README_dataset_hebrew.md
└── README_data_extraction.md
```

תיקייה זו מכילה את קובץ הנתונים המקורי ואת קבצי התיעוד המתארים את מקור הנתונים, מבנה הדאטהסט ומשמעות המשתנים.

---

## E Commerce Dataset.xlsx

קובץ האקסל המקורי שעליו מבוסס המחקר.

הקובץ כולל את נתוני הלקוחות ואת עמודת המטרה:

```text
Churn
```

כאשר:

<div dir="rtl" align="right">

• `1` – לקוח נטש.
• `0` – לקוח לא נטש.

</div>

עמודות שאינן משמשות כמאפייני קלט באימון המודלים:

<div dir="rtl" align="right">

• `CustomerID` – מזהה טכני של הלקוח.
• `Churn` – משתנה המטרה שאותו המודלים מנסים לחזות.

</div>

---

## README_dataset_hebrew.md

קובץ תיעוד בעברית עבור הדאטהסט.

מטרתו להסביר את מבנה הנתונים, משמעות המשתנים, עמודת המטרה והנחות העבודה המרכזיות שנלקחו בחשבון לפני בניית המודלים.

---

## README_data_extraction.md

קובץ תיעוד המתאר את מקור הנתונים ואת אופן השימוש בהם בפרויקט.

הקובץ מיועד להסביר כיצד הדאטהסט שולב בפרויקט ומהם שלבי ההכנה הראשוניים שנדרשו לפני האימון.

---

# תיקיית dummy_model

```text
dummy_model
│
├── dummy.py
├── dummy_baseline_summary.csv
└── README.md
```

תיקייה זו מכילה את מודל הבסיס הפשוט ביותר בפרויקט.

---

## dummy.py

סקריפט המפעיל מודל בסיסי מסוג Dummy Classifier.

מטרת המודל היא לספק נקודת ייחוס ראשונית לביצועי המודלים האחרים. כלומר, לבדוק מהו הביצוע שניתן לקבל ללא למידה אמיתית של דפוסים מתוך הנתונים.

מודל זה חשוב משום שהוא מאפשר לוודא שהמודלים המתקדמים אכן מוסיפים ערך מעבר לניחוש בסיסי.

---

## dummy_baseline_summary.csv

קובץ תוצאות המסכם את ביצועי מודל ה־Dummy.

הקובץ משמש כנקודת השוואה ראשונית מול Logistic Regression, Random Forest, XGBoost ו־CatBoost.

---

## README.md

קובץ תיעוד עבור תיקיית `dummy_model`.

מסביר את מטרת מודל הבסיס ואת תפקידו במסגרת המחקר.

---

# תיקיית LogisticRegression

```text
LogisticRegression
│
├── logistic_experiment_1_holdout.py
├── logistic_experiment_1_holdout_summary.csv
├── logistic_experiment_2_holdout.py
├── logistic_experiment_2_holdout_summary.csv
├── README.md
└── Old
```

תיקייה זו מכילה את ניסויי Logistic Regression, ששימשו כמודלים ליניאריים בסיסיים להשוואה מול מודלים מורכבים יותר.

---

## logistic_experiment_1_holdout.py

סקריפט ניסוי ראשון עבור Logistic Regression בגישת Holdout.

הסקריפט מבצע:

<div dir="rtl" align="right">

• טעינת הנתונים.
• חלוקת Train/Test.
• קדם־עיבוד למשתנים נומריים וקטגוריאליים.
• אימון מודל Logistic Regression.
• הערכת ביצועים על סט הבדיקה.

</div>

---

## logistic_experiment_1_holdout_summary.csv

קובץ סיכום התוצאות של `logistic_experiment_1_holdout.py`.

הקובץ כולל את מדדי הביצועים שהתקבלו בניסוי הראשון.

---

## logistic_experiment_2_holdout.py

סקריפט ניסוי שני עבור Logistic Regression.

ניסוי זה מהווה המשך או שיפור לניסוי הראשון, וכולל התאמות מתודולוגיות כגון שינויי קדם־עיבוד, איזון מחלקות או בדיקת תצורה משופרת של המודל.

---

## logistic_experiment_2_holdout_summary.csv

קובץ סיכום התוצאות של הניסוי השני ב־Logistic Regression.

משמש להשוואה בין שתי תצורות הניסוי של המודל הליניארי.

---

## README.md

קובץ תיעוד עבור תיקיית `LogisticRegression`.

מסביר את מטרת המודל, מבנה הניסויים והקשר שלו לשאר המודלים בפרויקט.

---

## תיקיית LogisticRegression/Old

```text
Old
│
├── logistic_experiment_1.py
├── logistic_experiment_1_summary.csv
├── logistic_experiment_2.py
├── logistic_experiment_2_summary.csv
└── README.md
```

תיקייה זו מכילה גרסאות ישנות יותר של ניסויי Logistic Regression.

היא נשמרה לצורך תיעוד היסטוריית הפיתוח וההשוואה בין גרסאות קוד קודמות לגרסאות ה־Holdout המעודכנות.

---

# תיקיית RandomForest

```text
RandomForest
│
├── Feature_Importance.py
├── random_forest_30_70.py
├── random_forest_experiment_1_summary.csv
├── random_forest_feature_importances.csv
├── random_forest_threshold.py
├── ReadMe.md
└── old
```

תיקייה זו מכילה את ניסויי Random Forest ואת ניתוח חשיבות המאפיינים של המודל.

---

## random_forest_30_70.py

סקריפט אימון והערכה של Random Forest בחלוקת Holdout של 70% אימון ו־30% בדיקה.

הסקריפט מבצע:

<div dir="rtl" align="right">

• טעינת הדאטהסט.
• הסרת `CustomerID` ו־`Churn` מהמאפיינים.
• קדם־עיבוד למשתנים חסרים.
• One-Hot Encoding למשתנים קטגוריאליים.
• אימון Random Forest.
• הערכת ביצועי המודל על סט הבדיקה.

</div>

---

## random_forest_threshold.py

סקריפט המיועד לבדיקת ערכי Threshold שונים עבור Random Forest.

במקום להשתמש רק בסף ברירת המחדל `0.5`, הסקריפט בוחן כיצד שינוי הסף משפיע על:

<div dir="rtl" align="right">

• Precision.
• Recall.
• F1 Score.
• זיהוי לקוחות נוטשים.
• איזון בין טעויות מסוגים שונים.

</div>

---

## Feature_Importance.py

סקריפט לחישוב חשיבות המאפיינים במודל Random Forest.

הסקריפט מאמן את המודל על כלל הדאטהסט ומפיק דירוג Feature Importance.

מאחר ש־Random Forest אינו עובד ישירות עם משתנים קטגוריאליים, מתבצע One-Hot Encoding. לאחר מכן, חשיבויות הקטגוריות שנוצרו מאותו משתנה מאוחדות בחזרה למאפיין המקורי.

לדוגמה:

```text
MaritalStatus_Single
MaritalStatus_Married
MaritalStatus_Divorced
```

מאוחדים חזרה ל:

```text
MaritalStatus
```

מטרת האיחוד היא לאפשר פרשנות ברמת 18 המאפיינים המקוריים ולהשוות את התוצאות מול CatBoost.

---

## random_forest_experiment_1_summary.csv

קובץ סיכום הביצועים של ניסוי Random Forest.

כולל את מדדי ההערכה המרכזיים של המודל.

---

## random_forest_feature_importances.csv

קובץ CSV המכיל את דירוג חשיבות המאפיינים של Random Forest.

הקובץ משמש לניתוח אילו משתנים השפיעו ביותר על החלטות המודל.

---

## ReadMe.md

קובץ תיעוד עבור תיקיית `RandomForest`.

מסביר את מטרת הניסויים, מבנה הקבצים ותפקיד המודל במסגרת המחקר.

---

## תיקיית RandomForest/old

```text
old
│
├── random_forest.py
└── random_forest_experiment_1_summary.csv
```

תיקייה זו מכילה גרסה ישנה יותר של ניסוי Random Forest.

היא נשמרה לצורך תיעוד התפתחות הקוד והשוואה מול הגרסאות המעודכנות.

---

# תיקיית XGBoost

```text
XGBoost
│
├── README.md
├── xgboost_experiment_1_holdout.py
├── xgboost_experiment_1_holdout_summary.csv
├── xgboost_experiment_2_holdout.py
├── xgboost_experiment_2_holdout_summary.csv
└── Old
```

תיקייה זו מכילה את ניסויי XGBoost, מודל Boosting מתקדם המבוסס על עצי החלטה.

---

## xgboost_experiment_1_holdout.py

סקריפט הניסוי הראשון של XGBoost בגישת Holdout.

הסקריפט כולל טעינת נתונים, קדם־עיבוד, אימון המודל והערכתו על סט הבדיקה.

---

## xgboost_experiment_1_holdout_summary.csv

קובץ סיכום התוצאות של הניסוי הראשון ב־XGBoost.

---

## xgboost_experiment_2_holdout.py

סקריפט ניסוי שני עבור XGBoost.

ניסוי זה בודק תצורה נוספת או משופרת של המודל, במטרה לבחון האם ניתן לשפר את יכולת החיזוי ביחס לניסוי הראשון.

---

## xgboost_experiment_2_holdout_summary.csv

קובץ סיכום התוצאות של הניסוי השני ב־XGBoost.

---

## README.md

קובץ תיעוד עבור תיקיית `XGBoost`.

מסביר את תפקיד המודל ואת מבנה הניסויים בתיקייה.

---

## תיקיית XGBoost/Old

```text
Old
│
├── README.md
├── xgboost_experiment_1.py
├── xgboost_experiment_1_summary.csv
├── xgboost_experiment_2.py
└── xgboost_experiment_2_summary.csv
```

תיקייה זו כוללת גרסאות ישנות יותר של ניסויי XGBoost.

היא נשמרה לצורך תיעוד גרסאות קוד קודמות לפני המעבר לניסויי Holdout המעודכנים.

---

# תיקיית catboost

```text
catboost
│
├── catboost_experiment_1.py
├── catboost_experiment_1_summary.csv
├── catboost_experiment_2.py
├── catboost_experiment_2_summary.csv
├── catboost_old1.py
├── catboost_old2.py
├── README.md
└── catboost_info
```

תיקייה זו מכילה את ניסויי CatBoost, מודל Boosting מתקדם המסוגל לעבוד ישירות עם משתנים קטגוריאליים.

---

## catboost_experiment_1.py

סקריפט הניסוי הראשון של CatBoost.

מטרתו לבחון את ביצועי CatBoost בתצורה ראשונית על הדאטהסט.

המודל מנצל את יכולתו לטפל במשתנים קטגוריאליים באופן מובנה, ללא צורך ב־One-Hot Encoding.

---

## catboost_experiment_1_summary.csv

קובץ סיכום התוצאות של הניסוי הראשון ב־CatBoost.

---

## catboost_experiment_2.py

סקריפט הניסוי השני של CatBoost.

ניסוי זה כולל תצורה מתקדמת יותר של המודל, לרבות שימוש באיזון מחלקות ובדיקת ערכי Threshold שונים.

מטרת הניסוי היא לשפר את זיהוי הלקוחות הנוטשים ולבחון את רגישות המודל לספי החלטה שונים.

---

## catboost_experiment_2_summary.csv

קובץ סיכום התוצאות של הניסוי השני ב־CatBoost.

הקובץ משמש להשוואת ביצועים בין ערכי Threshold שונים ובין ניסויי CatBoost השונים.

---

## catboost_old1.py

גרסה ישנה של סקריפט CatBoost.

נשמרה לצורך תיעוד שלבי הפיתוח והמעבר לגרסאות ניסוי מסודרות יותר.

---

## catboost_old2.py

גרסה ישנה נוספת של סקריפט CatBoost.

משמשת לשימור היסטוריית הניסויים וההתפתחות של הקוד.

---

## README.md

קובץ תיעוד עבור תיקיית `catboost`.

מסביר את מטרת ניסויי CatBoost ואת מבנה הקבצים בתיקייה.

---

## תיקיית catboost_info

```text
catboost_info
│
├── catboost_training.json
├── learn_error.tsv
├── time_left.tsv
└── learn
```

תיקייה זו נוצרת אוטומטית על ידי CatBoost בזמן האימון.

היא כוללת קבצי לוג ומעקב אחר תהליך הלמידה.

---

## catboost_training.json

קובץ JSON שנוצר על ידי CatBoost ומתעד פרטים על תהליך האימון.

יכול לכלול מידע על פרמטרים, מדדים ומצב האימון.

---

## learn_error.tsv

קובץ טבלאי המתעד את שגיאת הלמידה לאורך האיטרציות של המודל.

משמש למעקב אחר התקדמות האימון.

---

## time_left.tsv

קובץ המתעד מידע על זמן האימון והזמן המשוער שנותר.

---

## catboost_info/learn/events.out.tfevents

קובץ אירועים שנוצר לצורך מעקב אחר האימון, לרוב בפורמט המתאים להצגה בכלים כגון TensorBoard.

---

# סיכום מטרת הפרויקט

מטרת פרויקט `Churn_Customer` היא לחזות נטישת לקוחות בפלטפורמת מסחר אלקטרוני באמצעות מספר מודלים של למידת מכונה, ולהשוות ביניהם מבחינת ביצועים, יציבות ויכולת פרשנות.

המודלים שנבחנו בפרויקט:

<div dir="rtl" align="right">

• Dummy Model – מודל בסיס להשוואה.
• Logistic Regression – מודל ליניארי בסיסי.
• Random Forest – מודל אנסמבל מבוסס עצי החלטה.
• XGBoost – מודל Boosting מתקדם.
• CatBoost – מודל Boosting עם טיפול מובנה במשתנים קטגוריאליים.

</div>

ההשוואה בין המודלים מבוססת על מדדים כגון:

<div dir="rtl" align="right">

• Accuracy.
• Balanced Accuracy.
• Precision.
• Recall.
• F1 Score.
• F1 Macro.
• ROC AUC.
• PR AUC.
• Confusion Matrix.
• Feature Importance.

</div>

הפרויקט מאפשר לבחון לא רק איזה מודל מספק את הביצועים הטובים ביותר, אלא גם אילו מאפיינים משפיעים ביותר על חיזוי נטישת הלקוחות.

</div>
