# Data Extraction — E-Commerce Customer Churn Dataset

## Source

**Kaggle Dataset**: E-Commerce Customer Churn Analysis and Prediction  
**Author**: ankitverma2010  
**URL**: https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction

---

## Manual Download Steps

1. Go to the dataset page on Kaggle (URL above).
2. Click the **Download** button (top right of the page).
   - You must be logged in to a Kaggle account.
3. A file named **`archive.zip`** will be downloaded.
4. Extract the zip — it contains a single file: **`E Commerce.xlsx`**
5. Place the extracted file in:
   ```
   Churn_Customer/dataset/E Commerce.xlsx
   ```

### Expected folder structure after extraction

```
Churn_Customer/
└── dataset/
    └── E Commerce.xlsx
```

---

## Dependencies

Install required Python packages before running any experiment:

```bash
pip install pandas scikit-learn openpyxl
```

- `pandas` — data loading and manipulation
- `scikit-learn` — preprocessing, model, cross-validation
- `openpyxl` — reading `.xlsx` Excel files with pandas

---

## Running Experiment 1

```bash
cd Churn_Customer/LogisticRegression
python logistic_experiment_1.py
```

The script will:
1. Load `../dataset/E Commerce.xlsx` (Sheet: `E Comm`)
2. Drop the `CustomerID` column
3. Build a preprocessing pipeline (imputation + scaling/encoding)
4. Train a Logistic Regression model with 5-fold Stratified Cross-Validation
5. Print all metrics (Accuracy, F1 Macro, ROC AUC, Confusion Matrix)
6. Save results to `logistic_experiment_1_summary.csv`

---

## Why This Dataset

All previous experiments on the synthetic e-commerce dataset (`ecommerce_customer_behavior_dataset.csv`) achieved **ROC AUC ≈ 0.5** — essentially random. Investigation confirmed the `Is_Returning_Customer` label was randomly assigned with no statistical relationship to any feature.

This Kaggle dataset is **real data** with a genuine behavioral signal: customers who churned had measurably different usage patterns (tenure, order frequency, cashback usage, etc.) compared to those who stayed.
