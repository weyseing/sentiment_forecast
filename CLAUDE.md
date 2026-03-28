# CLAUDE.md — Project Context

## Project Overview

**Title:** Forecasting Mental Health Sentiment Surges: A Time-Series Analysis of Reddit Discourse
**Student:** Heng Wey Seing (ID: 24042426)
**Course:** MRP5015 Capstone Project 1 — Master of Data Science (Sunway University)
**Supervisor:** Prof. Dr Selina Low Yeh Ching

---

## Full Pipeline

### Goal
Predict when university students will be most stressed, using Reddit posts as data.

---

### Step 1: Collect Data (Reddit Scraper)
- Grab posts & comments from Reddit via **Arctic Shift API** (free, no API key, historical archive)
- **Subreddits in dataset:** r/college, r/Students, r/GradSchool, r/mentalhealth (student-context filtered)
- **Duration:** 2 years — 2024-01-15 to 2025-12-19 (705 days)
- **Two-stage filtering:**
  - Stage 1 (scraper): broad keyword filter (stress keywords); r/mentalhealth also requires student-context keyword match (e.g. "exam", "semester", "campus")
  - Stage 2 (NLP, Step 2): VADER + RoBERTa to confirm genuine stress
- **Output:** `data/2years/1_reddit_raw.csv` — 138,058 rows

---

### Step 2: NLP Classification — Is this post "stressed"? (Yes/No)

Each post classified as **stress = 1** or **not stress = 0** using hybrid of two models:

| Model | How it works | Strength | Weakness |
|-------|-------------|----------|----------|
| VADER | Dictionary lookup, scores -1 to +1 | Fast, explainable | Misses sarcasm & context |
| RoBERTa | AI transformer | Understands context | Slower, black box |

**Decision logic:**
- Both say STRESSED → 1 (high confidence)
- Both say NOT STRESSED → 0 (high confidence)
- They disagree → -1 (manual review by principal investigator)

**Output:** `data/2years/2_reddit_labeled.csv` — adds `vader_label`, `roberta_label`, `is_stressed`, `needs_review`

---

### Step 3: Daily Count Aggregation

Group classified posts by date into a daily time series of stressed post counts.

- Days with zero total_posts are dropped (scraper cut-off artifacts)
- **Output:** `data/2years/3_daily_counts.csv` — 705 rows
- Key stats: mean=100.4, max=199, min=33 stressed posts/day

---

### Step 4: Statistical Modelling — What drives the stress count?

Fit a GLM to explain WHY daily counts go up or down:

```
stressed ~ week_number + C(day_of_week_name, Treatment('Monday'))
```

Compare two models:
- **Poisson** — assumes variance = mean (too simple for social media spikes)
- **Negative Binomial** — handles overdispersion (winner, ΔAIC = 2,381)

**Key results:**
- Overdispersion ratio = 7.33 — NB justified
- Weekend effect: Sat -18%, Sun -15%, Fri -11% vs Monday (all significant)
- Mon–Thu: no significant difference
- Upward trend: +0.3%/week = +37% over 2 years

**Output:** `data/2years/4_irr_table.csv`, `4_model_comparison.csv`, `4_residuals.png`

---

### Step 5: Forecasting — Predict future stress surges

Walk-forward validation (expanding window, 4 windows, horizon=21 days):

| Model | Type | Result |
|-------|------|--------|
| Prophet | Facebook, yearly + weekly seasonality, multiplicative | Mean MAE = 32.01 |
| SARIMA(1,1,1)(1,1,1,7) | Seasonal ARIMA, weekly period=7 | Mean MAE = 15.57 (Winner) |

- ARIMA(1,1,1) was tested first but produced flat forecasts — replaced by SARIMA
- Prophet beats SARIMA on Window 4 (MAE 10.7 vs 12.5) when trained on full 2-year data
- Prophet needs at least 1 full year of data to learn yearly seasonality

**Output:** `data/2years/5_cv_scores.csv`, `5_cv_summary.csv`, `5_cv_plot.png`, `5_final_forecast.csv`, `5_final_forecast.png`

---

### Full Pipeline Summary

```
Reddit Posts (4 subreddits, 2 years)
    ↓
[src/1_scrape_reddit.py] → data/2years/1_reddit_raw.csv (138,058 rows)
    ↓
[src/2_classify_sentiment.py] → data/2years/2_reddit_labeled.csv
    ↓
[src/3_aggregate_counts.py] → data/2years/3_daily_counts.csv (705 days)
    ↓
[src/4_model_glm.py] → NB GLM: weekend effect + upward trend confirmed
    ↓
[src/5_forecast.py] → SARIMA wins overall; Prophet competitive with full data
```

---

## Scripts

| File | Purpose |
|------|---------|
| `src/1_scrape_reddit.py` | Reddit data collection via Arctic Shift API |
| `src/2_classify_sentiment.py` | VADER + RoBERTa hybrid classification |
| `src/3_aggregate_counts.py` | Daily stress count aggregation |
| `src/4_model_glm.py` | Poisson vs Negative Binomial GLM |
| `src/5_forecast.py` | Prophet vs SARIMA walk-forward forecasting |
| `src/check_dataset.py` | Dataset completeness and quality checks |

## Data

| File | Description |
|------|-------------|
| `data/2years/1_reddit_raw.csv` | Step 1 output — 138,058 raw posts (2024-2025) |
| `data/2years/2_reddit_labeled.csv` | Step 2 output — adds is_stressed labels |
| `data/2years/3_daily_counts.csv` | Step 3 output — 705-day daily time series |
| `data/2years/4_irr_table.csv` | Step 4 output — IRRs from NB model |
| `data/2years/4_model_comparison.csv` | Step 4 output — AIC/BIC comparison |
| `data/2years/4_residuals.png` | Step 4 output — residual diagnostic plots |
| `data/2years/5_cv_scores.csv` | Step 5 output — MAE/RMSE/MAPE per window |
| `data/2years/5_cv_summary.csv` | Step 5 output — mean scores across windows |
| `data/2years/5_cv_plot.png` | Step 5 output — walk-forward validation plot |
| `data/2years/5_final_forecast.csv` | Step 5 output — 21-day Prophet forecast |
| `data/2years/5_final_forecast.png` | Step 5 output — final forecast chart |
| `data/1sem/` | 1-semester dataset (subset, earlier analysis) |

## Docs

| File | Description |
|------|-------------|
| `docs/step_report.html` | A4 Word-style report covering Steps 1-5 with tables and charts |

## Assessments

| File | Description |
|------|-------------|
| `assessments/ASSESSMENT_1_ACTIVITY_LOG.md` | Reflective writing + supervision meeting records |
| `assessments/ASSESSMENT_2_LITERATURE_REVIEW.md` | Full literature review |
| `assessments/ASSESSMENT_3_RESEARCH_PROPOSAL.md` | Detailed methodology & research proposal |
