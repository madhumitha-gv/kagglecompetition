# Triagegeist — Emergency Triage Acuity Prediction

A machine learning system that predicts Emergency Severity Index (ESI) triage levels from structured patient intake data, augmented with NLP-extracted chief complaint features and demographic bias analysis.

## Problem Statement

Emergency department nurses manually assign ESI scores (1-5) to incoming patients under extreme cognitive load. Errors lead to delayed care and preventable deaths. This project builds a clinical decision support model to assist triage decisions.

## Dataset

Synthetic ED dataset by the Laitinen-Fredriksson Foundation — 80,000 training records across 4 files:

| File | Description |
|---|---|
| train.csv | Vitals, demographics, triage context |
| patient_history.csv | 25 binary comorbidity flags |
| chief_complaints.csv | Free-text complaint narratives |
| test.csv | Test set without target |

**Target:** `triage_acuity` — ESI level 1 (critical) to 5 (minor)

## Approach

### 1. Preprocessing
- Merged 4 files on `patient_id`
- Excluded post-triage leakage columns (`disposition`, `ed_los_hours`)
- Created missingness flags for vitals (missing BP = lower acuity signal)
- Fixed `pain_score = -1` encoding

### 2. Feature Engineering
- `news2_x_gcs` — NEWS2 score × (16 - GCS), capturing combined physiological deterioration
- `hr_spo2` — heart rate × oxygen deficit, flagging tachycardic hypoxic patients
- Missingness flags: `bp_missing`, `resp_missing`, `temp_missing`

### 3. NLP Features
- 10 clinical keyword flags from chief complaint text (`has_chest_pain`, `has_bleeding`, `has_stroke` etc.)
- Severity word extraction from complaint structure (`severity_severe`, `severity_moderate`, `severity_mild`)
- TF-IDF was excluded after detecting synthetic data leakage (4,934/4,949 unique complaints map to single acuity level)

### 4. Model
- XGBoost classifier with 5-fold stratified cross-validation
- Handles missing vitals natively (no imputation needed)

### 5. Bias Analysis
- Audited undertriage rates across language, insurance type, and sex groups
- Arabic speakers showed highest undertriage rate (4.9%) vs Estonian speakers (4.0%)

## Results

| Model | CV Accuracy |
|---|---|
| Baseline vitals + history | 85.71% |
| + Clinical keyword flags | 87.15% |
| + Severity words from text | 89.25% |

**Top features:** `news2_x_gcs`, `news2_score`, `gcs_total`, `pain_score`, `severity_severe`

## Key Findings

- `mental_status_triage` is the strongest categorical predictor — 42.6% of unresponsive patients are ESI 1
- Missing vitals carry independent signal — BP missingness occurs almost exclusively in ESI 4/5 patients
- Pain location and arrival mode show negligible differences across acuity levels
- Severity words in complaint text (`severe`, `moderate`, `mild`) are strongly predictive and leak-free

## How To Run

```bash
# Clone the repo
git clone https://github.com/yourusername/triagegeist

# Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib

# Run notebook
jupyter notebook triagegeist.ipynb
```

## Competition

Kaggle — Triagegeist: AI in Emergency Triage
$10,000 prize pool — Laitinen-Fredriksson Foundation

## Tools & Libraries

Python, XGBoost, scikit-learn, pandas, numpy, matplotlib

## Author

Built as part of Kaggle competition participation — April 2026
