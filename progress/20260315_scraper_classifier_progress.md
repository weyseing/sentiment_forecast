# Progress Log — 2026-03-15

## Step 1: Reddit Scraper

### Completed
- Renamed `src/scrape_reddit.py` → `src/1_scrape_reddit.py`
- Added 7 new student-specific subreddits: r/GradSchool, r/AskAcademia, r/learnprogramming, r/premed, r/lawschool, r/nursing, r/EngineeringStudents
- Added student-context keyword filter on r/mentalhealth to avoid general-population noise (was 85% of prior dataset)
- Output file updated to `data/1_reddit_raw.csv`

### Prior Scrape (backup)
- `data/1_reddit_raw_1.csv` — 35,771 rows, Fall 2025 (Sep 1–Dec 20)
- Subreddits: r/college, r/students, r/mentalhealth only
- Issues: r/mentalhealth dominated at 85%; Dec 21 missing; r/Students missing Sep 24 & Oct 17

### Re-scrape
- In progress — running with expanded subreddits + student-context filter

---

## Step 2: NLP Sentiment Classifier

### Completed
- Created `src/2_classify_sentiment.py`
- VADER + RoBERTa hybrid (`cardiffnlp/twitter-roberta-base-sentiment-latest`)
- Checkpoint/resume support — safe to interrupt, appends batch by batch
- CLI args: `--input` / `--output` / `--help`
- Output: `data/2_reddit_labeled.csv` — adds `vader_label`, `roberta_label`, `is_stressed`, `needs_review`

### In Progress
- Running classifier on `data/1_reddit_raw_1.csv` (prior scrape) as test run
- ~35k rows, CPU only, estimated ~22 minutes

---

## Next Steps
- [ ] Step 3: `3_aggregate_counts.py` — daily stress count time series
- [ ] Step 4: `4_model_glm.py` — Poisson / Negative Binomial regression
- [ ] Step 5: `5_forecast.py` — Prophet + ARIMA forecasting
