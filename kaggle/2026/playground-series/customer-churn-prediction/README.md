# Customer Churn Prediction

> Identifying at-risk telecom customers before they leave —
> enabling proactive retention over reactive damage control.

![Model](https://img.shields.io/badge/Model-LightGBM-orange)
![Metric](https://img.shields.io/badge/Metric-ROC--AUC-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Business Problem

**Customer churn** is the rate at which customers stop doing 
business with a company over a given period. In the telecom 
industry, where switching costs are low and competition is high, 
churn is one of the most critical metrics to monitor and minimize.

Retaining an existing customer costs significantly less than 
acquiring a new one — yet most retention efforts are reactive, 
triggered only after a customer has already decided to leave. 
By that point, the opportunity to intervene is often lost.

This project takes a proactive approach: building a machine 
learning model that scores every customer by their likelihood 
of churning — giving retention teams the intelligence to 
prioritize outreach, personalize offers, and act before 
revenue walks out the door.

**Key business questions this model answers:**
- Which customers are at highest risk of churning this month?
- What behavioral and contractual signals drive churn?
- How can the business segment customers by churn risk 
  to allocate retention spend efficiently?

---

## 📦 Dataset

| Attribute | Detail |
|-----------|--------|
| Training rows | 594,194 customers |
| Test rows | 254,655 customers |
| Features | 19 (15 categorical, 4 continuous) |
| Target | Churn (Yes / No) |
| Class distribution | 77.5% retained / 22.5% churned |
| Evaluation metric | ROC-AUC Score |

---

## 🔍 What the Data Revealed

| Feature | Business Insight |
|---------|-----------------|
| Tenure | Churners leave early — avg 17 months vs 42 months for loyal customers |
| Monthly Charges | Churners pay $20 more per month on average |
| Contract Type | Month-to-month customers churn at 42% vs 3% on 2-year contracts |
| High Risk Flag | New customers on high charges churn 8.6x more than the average |

---

## 🛠️ Approach

### Step 1 — Exploratory Data Analysis
- Identified class imbalance (77/22 split)
- Detected heavy right skew in TotalCharges
- Analyzed churn rate across all categorical segments
- Identified high risk customer profile as strongest signal

### Step 2 — Feature Engineering

| Feature | Business Logic |
|---------|---------------|
| NumServices | Customers using more services have higher switching cost |
| AvgMonthlySpend | Historical average vs current charges reveals price drift |
| ChargeIncrease | Rising charges signal potential dissatisfaction |
| TenureGroup | Segments customers by loyalty stage |
| HighRiskFlag | New customer + high charges = highest churn risk |
| ChargePerService | Value for money — high cost per service signals churn |

### Step 3 — Modeling
- Log transform applied to fix TotalCharges skewness
- LightGBM with 5-Fold Stratified Cross Validation
- GPU accelerated training
- Class weight balancing for imbalanced target
- Early stopping monitored directly on AUC

### Step 4 — Ensemble
- Trained multiple LightGBM model variants
- Hill Climbing algorithm to find optimal blend weights
- Searched 2000 weight combinations automatically
- Final weights selected based on out-of-fold AUC

---

## 📊 Results

| Model | AUC Score |
|-------|-----------|
| Baseline | 0.9134 |
| Tuned LightGBM | 0.9164 |
| Final Ensemble | 0.9136 |

---

## 💡 Key Learnings

- **Match your metric** — evaluation metric must align with 
  business objective. Using logloss instead of AUC as early 
  stopping criterion cost 0.008 in model performance
- **Tree models are scale invariant** — gradient boosting 
  splits on thresholds, not distances, so normalization 
  is unnecessary
- **Diversity beats complexity** — different model 
  architectures ensemble better than similar ones
- **Class imbalance handling** — scale_pos_weight more 
  effective than SMOTE for tree based models on this data
- **Save intermediate predictions** — out-of-fold predictions 
  must be persisted to disk; session resets lose all progress

---

## 🔄 Further Improvements

- **CatBoost** — handles categorical features natively 
  without label encoding, preserving true category relationships
- **Diverse ensemble** — XGBoost + LightGBM + CatBoost 
  trained on different encodings for maximum prediction diversity
- **Target encoding** — replace label encoding with churn 
  rate per category for more meaningful feature representation
- **Neural networks** — TabNet or MLP on tabular data 
  as a complementary model type
- **Hyperparameter tuning** — systematic search with Optuna 
  rather than manual parameter selection

---

## 🧰 Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.x-blue)
