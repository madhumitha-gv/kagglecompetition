# March Machine Learning Mania 2026
> Forecasting NCAA basketball tournament outcomes 
> using 40 years of game history and machine learning.

![Model](https://img.shields.io/badge/Model-XGBoost-red)
![Metric](https://img.shields.io/badge/Metric-Brier%20Score-blue)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Business Problem

Every March, 68 college basketball teams compete in the 
NCAA tournament — and millions of fans attempt to predict 
the outcomes. Most predictions rely on gut feeling and 
surface-level statistics.

This project takes a data-driven approach: building a 
machine learning model that assigns a win probability 
to every possible matchup between Division I teams. 
Rather than predicting a single bracket, the model 
outputs calibrated probabilities for all 132,133 
possible team pairings — men's and women's combined.

Key questions this model answers:
- Given any two teams, what is the probability one beats the other?
- How does historical team strength, seeding, and expert rankings inform outcomes?
- How do men's and women's tournaments differ in predictability?

---

## 📦 Dataset

| Attribute | Detail |
|-----------|--------|
| Men's regular season games | 198,577 games |
| Women's regular season games | 142,507 games |
| Men's tournament games | 2,585 games |
| Women's tournament games | 1,717 games |
| Historical coverage | Men's: 1985–2026, Women's: 1998–2026 |
| Total predictions required | 132,133 matchups |
| Evaluation metric | Brier Score (lower is better) |

---

## 🔍 What the Data Revealed

| Insight | Finding |
|---------|---------|
| Seed 1 vs 16 | #1 seed wins 98.8% of the time historically |
| Seed 8 vs 9 | Essentially a coin flip — 9 seed wins slightly more |
| Seed 5 vs 12 | Famous upset zone — only 64.4% for the 5 seed |
| Women's model | More predictable than men's due to dominant programs |
| Win% alone | Misleading — weak schedule inflates rankings |

---

## 🛠️ Approach

### Step 1 — Elo Rating System
Built a custom Elo rating from scratch using all historical game results:
- Every team starts at 1500
- Ratings update after every game based on opponent strength
- Winning vs strong opponent = big Elo gain
- 25% mean reversion applied each new season

### Step 2 — Season Statistics
Computed per-team per-season features:
- Win percentage
- Average points scored and allowed
- Point differential

### Step 3 — Massey Ordinals
Integrated expert ranking systems (POM, MOR, RPI):
- Composite score across 3 systems
- Pre-tournament rankings only to avoid data leakage
- Available for men's data only

### Step 4 — Tournament Seeds
Extracted historical seed win rates:
- Seeds are the strongest single predictor
- Correlation with outcome: 0.49

### Step 5 — Modeling
Feature differences between two teams for every historical matchup:

| Feature | Correlation with Outcome |
|---------|--------------------------|
| SeedDiff | 0.492 |
| MasseyDiff | 0.344 |
| EloDiff | 0.340 |
| PointDiff | 0.335 |
| WinPctDiff | 0.324 |

Trained ensemble of Logistic Regression and XGBoost
on historical tournament games with walk-forward cross validation.

---

## 📊 Results

| Model | Brier Score |
|-------|-------------|
| Random baseline | 0.2500 |
| Logistic Regression | 0.1873 |
| XGBoost ensemble | ~0.181 |
| Women's model | 0.1421 |

Brier Score measures calibration of probabilities.
Lower is better. Random guessing scores 0.25.

---

## 💡 Key Learnings

- **Elo beats win%** — schedule strength matters; a team going 
  28-1 in a weak conference is not the same as 28-1 in a power conference
- **Seeds are powerful** — the selection committee does significant 
  work; seed difference alone explains nearly 50% of outcomes
- **Women's more predictable** — dominant programs like UConn create 
  clearer separations between strong and weak teams
- **Walk-forward validation is essential** — training and testing on 
  same years inflates performance metrics significantly
- **2021 was an outlier** — COVID bubble tournament produced unusual 
  results no model could predict

---

## 🔄 Further Improvements

- **Margin of victory in Elo** — weight wins by score differential not just result
- **Box score features** — pace, offensive and defensive efficiency
- **Recent seasons weighted higher** — 2026 basketball differs from 1985
- **Separate models per round** — early round dynamics differ from Final Four
- **Home court adjustment** — regular season results include home/away games

---

## 🧰 Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.x-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.x-lightblue)
