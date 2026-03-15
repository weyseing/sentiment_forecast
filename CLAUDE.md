# CLAUDE.md — Project Context

## Project Overview

**Title:** Forecasting Mental Health Sentiment Surges: A Time-Series Analysis of Reddit Discourse
**Student:** Heng Wey Seing (ID: 24042426)
**Course:** MRP5015 Capstone Project 1 — Master of Data Science (Sunway University)
**Supervisor:** Prof. Dr Selina Low Yeh Ching

---

## Full Pipeline Explained Simply

### Goal
Predict when university students will be most stressed, using Reddit posts as data.

---

### Step 1: Collect Data (Reddit Scraper)
- Grab posts & comments from Reddit via **Arctic Shift API** (free, no API key, historical archive)
- **Subreddits:** r/college, r/students, r/mentalhealth
- **Duration:** 16 weeks (one full academic semester = 112 days), default Fall 2025: Sep 1 → Dec 21
- **Two-stage filtering:**
  - Stage 1 (scraper): broad keyword filter (30 stress keywords) to narrow ~50k posts → ~10-15k candidates
  - Stage 2 (NLP, next step): VADER + RoBERTa to confirm which candidates are genuinely stress-related
- Appends to CSV page-by-page — safe to interrupt and resume
- **Output:** `data/reddit_raw.csv` — one row per post/comment, with text + timestamp

---

### Step 2: NLP — Is this post "stressed"? (Yes/No)

Each post gets classified as **stress = 1** or **not stress = 0** using two models:

| Model | How it works | Strength | Weakness |
|-------|-------------|----------|----------|
| VADER | Dictionary lookup, scores -1 to +1 | Fast, explainable | Misses sarcasm & context |
| RoBERTa | AI transformer, trained on text | Understands context | Slower, black box |

**Decision logic:**
- Both say STRESSED → Stressed (high confidence)
- Both say NOT STRESSED → Not stressed (high confidence)
- They disagree (~15-20% of posts) → Manual review by principal investigator

**Output:** Same CSV + a new column `is_stressed = 1 or 0`

---

### Step 3: Count Aggregation — Numbers per Day

Take all the Yes/No labels and count them up by day:

```
Date         | Stress Posts
-------------|-------------
2025-03-01   |  12
2025-03-02   |  8
2025-03-03   |  45   ← exam day spike!
...
```

- **Output:** 112 rows (16 weeks × 7 days), one column = daily stress count

---

### Step 4: Statistical Modeling — What drives the stress count?

Fit a regression to explain WHY counts go up or down:

```
stress_count ~ days_until_exam + days_after_exam + day_of_week + week_number
```

Compare two models:
- **Poisson** — assumes variance ≈ mean (probably too simple for social media spikes)
- **Negative Binomial** — handles overdispersion/spikes better (likely winner)

Pick winner using AIC/BIC score (lower = better).

**Output:** Incidence rate ratios (IRRs) — e.g. "3 days before exam = 2× more stress posts"

---

### Step 5: Forecasting — Predict future stress surges

Feed the 112-day count series into forecasting models:

| Model | Type | Priority |
|-------|------|----------|
| Prophet | Facebook's tool, handles academic calendar events | Primary (essential) |
| ARIMA | Classical stats baseline | Comparative |
| LSTM | Deep learning, non-linear patterns | Optional (if time permits) |

- Predict **2–4 weeks ahead** with confidence intervals
- Evaluate on **last 3 weeks** (held out, never trained on) using MAE, RMSE, MAPE

**Output:** Forecast charts — e.g. "Week 12 will be high stress — finals period"

---

### Full Pipeline Summary

```
Reddit Posts
    ↓
[Scraper] → raw CSV
    ↓
[VADER + RoBERTa hybrid] → is_stressed column (1/0)
    ↓
[Daily Count Aggregation] → 112-day time series
    ↓
[Poisson / Negative Binomial GLM] → understand drivers
    ↓
[Prophet / ARIMA / LSTM] → predict future surges
    ↓
"Week 14 will have 40% more stress posts than average"
```

---

## Scripts

| File | Purpose |
|------|---------|
| `src/scrape_reddit.py` | Reddit data collection via Arctic Shift API (no API key needed) |

## Data

| File | Description |
|------|-------------|
| `data/reddit_raw.csv` | Raw scraped posts/comments (keyword-filtered candidates) |

## Assessments

| File | Description |
|------|-------------|
| `assessments/ASSESSMENT_1_ACTIVITY_LOG.md` | Reflective writing + supervision meeting records |
| `assessments/ASSESSMENT_2_LITERATURE_REVIEW.md` | Full literature review |
| `assessments/ASSESSMENT_3_RESEARCH_PROPOSAL.md` | Detailed methodology & research proposal |
