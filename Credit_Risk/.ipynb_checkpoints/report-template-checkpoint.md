# 📊 Credit Risk Analysis Using Machine Learning

## Overview of the Analysis

In this analysis, we built a machine learning model to predict loan status — specifically, whether a loan is **healthy (0)** or **high-risk (1)** — using financial features of borrowers.

### 🔍 Purpose

The goal was to create a predictive model that enables financial institutions to identify high-risk loans ahead of time and mitigate potential financial losses.

### 🧾 Dataset Description

The dataset, `lending_data.csv`, contains **77,536** loan records with the following features:

- `loan_size`: Amount of the loan  
- `interest_rate`: Interest rate applied to the loan  
- `borrower_income`: Annual income of the borrower  
- `debt_to_income`: Ratio of total debt to income  
- `num_of_accounts`: Number of financial accounts the borrower holds  
- `derogatory_marks`: Number of negative credit marks  
- `total_debt`: Total outstanding debt  
- `loan_status`: Target variable  
  - `0`: Healthy loan  
  - `1`: High-risk loan  

### 🎯 Target Variable Overview

```python
print(df["loan_status"].value_counts())

- 0 (Healthy loans): Majority class

- 1 (High-risk loans): Minority class

🧠 Machine Learning Process
1. Data Preprocessing

  - Features (X) and labels (y) were created by separating loan_status from the rest of the dataset.

2. Train/Test Split

  - 80% training, 20% testing using train_test_split.

3. Model Selection

  - Logistic Regression (LogisticRegression from sklearn.linear_model).

4. Model Training

  - Fitted on training data.

5. Model Evaluation

  - Performance assessed using:

    - Accuracy

    - Confusion Matrix

    - Precision

    - Recall

    - F1-score

✅ Results
Machine Learning Model: Logistic Regression
Accuracy: 0.99

📉 Confusion Matrix
[[14924    77]
 [   31   476]]

🎯 Precision
  - Healthy loans (0): 1.00

  - High-risk loans (1): 0.86

🎯 Recall
  - Healthy loans (0): 0.99

  - High-risk loans (1): 0.94

🎯 F1-Score
  - Healthy loans (0): 1.00

  - High-risk loans (1): 0.90

📌 Summary
The logistic regression model performs exceptionally well, achieving 99% overall accuracy.

  - It accurately predicts healthy loans nearly perfectly.

  - It performs strongly on high-risk loans:

    - Recall: 0.94 → It correctly identifies 94% of actual high-risk loans.

    - Precision: 0.86 → When the model predicts a loan as high-risk, it is correct 86% of the time.

✅ Recommendation
We recommend using this logistic regression model if the goal is:

  - High accuracy overall

  - Reliable detection of high-risk loans

