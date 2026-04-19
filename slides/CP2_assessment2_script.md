# Presentation Script — CP2 Assessment 2
**Forecasting Mental Health Sentiment Surges**
Target: ~12 minutes · 13 slides

---

## Slide 1 — Title (25 sec)

Good morning, everyone.

My name is Heng Wey Seing, student ID 24042426. My supervisor is Prof. Dr Selina Low Yeh Ching.

Today I present my Capstone Project 2: **"Forecasting Mental Health Sentiment Surges — A Time-Series Analysis of Reddit Discourse."**

The main question is simple: **can we predict when university students will be most stressed, just by reading their Reddit posts?**

---

## Slide 2 — Background & Research Question (1 min 10 sec)

Let me start with the problem.

Studies by Auerbach in 2018 and Lipson in 2022 show that **one in three university students** has anxiety or depression.

But universities measure this with surveys. Surveys are slow and expensive. They run once or twice a year. By the time the data comes back, the exam is already over.

Reddit is different. Students post **openly, anonymously, in real time**. So Reddit could be an **early warning signal** — if we read it correctly.

When I checked past studies, I found **three gaps**:

**First — the integration gap.** Most studies stop at labelling posts. Few go further to predict what comes next.

**Second — the classification gap.** Studies use VADER **or** RoBERTa, not both. When the two tools disagree, most studies just throw those posts away without saying so.

**Third — the time-coverage gap.** Most datasets cover less than one academic year. Daily patterns over two full years have not been studied.

So my research question: **Can a hybrid NLP and time-series pipeline, using Reddit posts, predict stress surges early enough for universities to act before they hit?**

---

## Slide 3 — Research Objectives (1 min)

I set three objectives — one for each gap.

**Objective 1 — the classification gap.** Label Reddit posts using **two models, VADER and RoBERTa**. When both agree, the label is high confidence. When they disagree, the post is excluded and reported openly.

**Objective 2 — the time-coverage gap.** Find **what drives daily stress counts over two years**. I fit a Negative Binomial regression on a 705-day time series to measure day-of-week, exam, and break effects. I report results as **percentage change**, so non-statisticians can read them.

**Objective 3 — the integration gap.** Predict stress surges **21 days ahead** using Prophet and SARIMA. I compare them with walk-forward validation and pick the winner by mean error.

Each objective produces a clear, testable output at a specific stage.

---

## Slide 4 — Methodology, Stages 1 and 2 (1 min)

These two stages handle data and labelling.

**Stage 1 — Collection.** I used the **Arctic Shift API** — free, no API key, no rate limit. I pulled posts from four subreddits — **r/college, r/Students, r/GradSchool, and r/mentalhealth** — from January 2024 to December 2025. I used 30 stress keywords to catch relevant posts. For r/mentalhealth, posts also had to match a student word like "exam" or "campus". In total: **138,058 posts**.

**Stage 2 — Classification.** Each post is checked by **two models**:

- **VADER** — a dictionary tool. Fast, but misses sarcasm.
- **RoBERTa** — a deep learning model. Slower, but understands context.

The main idea: I use **agreement as a confidence signal**. Both say stressed → label 1. Both say not stressed → label 0. They disagree → excluded. I originally planned to review disagreements by hand, but 40,500 cases were too many for one person. The agreement set was already big enough, so I dropped them and reported this openly.

---

## Slide 5 — Methodology, Stages 3, 4, and 5 (1 min)

These three stages turn labels into insight.

**Stage 3 — Aggregate.** I group posts by date to build a **705-day time series**. Days with zero posts are dropped because those are scraper gaps, not real silent days. I build **10 features** for modelling, including day of the week, exam flag, break flag, and subreddit share.

**Stage 4 — Explain.** I fit a **Poisson GLM** and a **Negative Binomial GLM**, and compare them by AIC. Social media counts are usually **overdispersed** — variance much bigger than the mean — and Negative Binomial handles that correctly. I report results as **incidence rate ratios**, which read as percentage change.

**Stage 5 — Predict.** I test **Prophet and SARIMA** with walk-forward validation: four expanding windows, 21 days ahead each time, always on unseen data. Both models get the same academic calendar events, so the test is fair.

Why separate the GLM and the forecast? They answer different questions. **The GLM tells us *why* counts move.** **The forecast tells us *when* the next surge hits.** Universities need both.

---

## Slide 6 — Tools (35 sec)

A quick word on tools.

- **Arctic Shift API** — free, no key, historical archive.
- **VADER + RoBERTa** — hybrid classifier.
- **statsmodels** — for the GLM.
- **Facebook Prophet** — for yearly and holiday effects.
- **SARIMA** — for the weekly pattern.

Every tool is open source and free. Any university can run this pipeline.

---

## Slide 7 — Findings: NLP Classification (1 min)

Let me show what the NLP stage produced.

Out of **138,058 posts**, the two models **agreed on 70.6%** — that is **97,527 high-confidence labels**. Of those, **51.3% were stressed** and **19.4% not stressed**.

The other **29.4% — about 40,500 posts** — were disagreements. These are the unclear, sarcastic, or context-heavy posts where a dictionary tool and a deep learning model see things differently.

I originally planned to review them by hand. But 40,500 was too many for one researcher. The 97,527 agreement cases were already a large, clean dataset, so I excluded the disagreements.

Why this matters:

**First — transparency.** Past studies often drop disagreements silently. My pipeline reports the exclusion openly and says exactly how many were dropped.

**Second — quality.** Keeping only cases where both models agree gives more trustworthy labels than any single-model pipeline.

---

## Slide 8 — Findings: GLM Results (1 min 30 sec)

This is the most important slide. It answers: **what drives daily stress counts?**

First, which model wins. Negative Binomial beats Poisson clearly. The AIC gap is **550 points** and the overdispersion ratio is **7.33** — variance is seven times the mean. Poisson cannot handle that, so Negative Binomial wins.

Now the results — as **incidence rate ratios**, or percentage change. Monday is the reference.

**Day of the week:**
- **Saturday — 18% fewer stress posts** than Monday.
- Sunday — 12.7% fewer.
- Friday — 12.3% fewer.
- All three: highly significant, p less than 0.001.
- Monday to Thursday — no real difference.

Meaning: students **switch off from school topics on weekends**.

**Calendar effects:**
- **Exam periods — up by 7.2%.** Significant.
- **Semester breaks — down by 11.1%.** Significant.

**Subreddit mix:** when r/mentalhealth share goes up, academic stress posts drop by 44.5%. That makes sense — r/mentalhealth is broader than school stress.

The most important finding is a **negative one**: week number is **not significant**, p equals 0.81. Over two years, there is **no upward trend**. Stress is cyclical — it follows the calendar — it is **not** getting worse each year.

---

## Slide 9 — Findings: Forecasting (1 min 15 sec)

Now the forecast. This answers **when** the next surge hits.

I compared **Prophet and SARIMA** with **walk-forward validation**. I start with a small training window, forecast the next 21 days, expand the window, forecast again. Four windows total. Always on unseen data.

**Overall winner: SARIMA.** Mean error of **21.61**, vs Prophet's **35.67**. SARIMA is off by about 22 posts per day on a series that averages 100.

But the per-window detail is more interesting:
- **Window 1 — SARIMA 28.4, Prophet 96.4.** Prophet struggles with less than a year of data.
- **Window 2 — SARIMA 21.9, Prophet 18.2.** Prophet catches up.
- **Window 3 — SARIMA 14.2, Prophet 18.4.** Both strong.
- **Window 4 — SARIMA 22.0, Prophet 9.8.** Prophet wins by a lot.

The rule is clear. **SARIMA is safer with short data** — it captures the weekly pattern directly. **Prophet wins once it sees a full year** — only then can it learn the yearly cycle.

So model choice depends on **how much data you have**.

---

## Slide 10 — Final Forecast Chart (45 sec)

This chart shows the full two-year series. **Green shading is semester breaks.** **Red shading is exam periods.** The right end is the **21-day forward forecast**.

You can see with your own eyes what the GLM confirmed. Counts **drop during breaks**, **rise during exams**, and **dip every weekend**. The forecast also has a **95% confidence band** — narrow means the model is confident, wide means be careful.

This is the **dashboard view** a student-affairs team would check each week.

---

## Slide 11 — Discussion (1 min 10 sec)

So what do these findings mean?

**First — the calendar effects.** Students post less on weekends, more during exams. This matches existing theory. What is new is the **daily level** and the **clear percentage numbers**. Past work only gave weekly averages, and only directions — not sizes.

**Second — the model-choice finding.** SARIMA wins short-term, Prophet wins long-term. A simple rule: one year of data → use SARIMA. Two years or more → switch to Prophet.

**Third — the no-trend finding.** Stress did **not** grow over two years. The problem is **cyclical, not worsening**. The right response is **timed support during exams**, not hiring more permanent staff.

Now the **limitations** — three honest points.

**Limitation 1:** The keyword filter leans toward stressed content. On purpose — cast a wide net first, then Stage 2 confirms.

**Limitation 2:** I dropped the 29.4% disagreement posts. Manual review was planned but not feasible at 40,500 cases. Clean labels, but lost signal.

**Limitation 3:** Reddit users are not all students. The results show what Reddit users say, not how all students feel.

---

## Slide 12 — Contributions (1 min 15 sec)

Three contributions that go beyond past work.

**First — the first full, end-to-end pipeline.** Past work either stops at classification or starts at forecasting with ready-made labels. No one has joined all five stages — scraping, hybrid NLP, aggregation, regression, and walk-forward forecasting — in one reproducible system. My pipeline closes that loop on 138,058 posts. All scripts are open source.

**Second — a concrete number for how much data Prophet needs.** Taylor and Letham said in 2018 that Prophet needs "sufficient" data, but gave no number. My walk-forward results give one. **At about two years of daily data, Prophet's Window 4 error drops to 9.8**, beating SARIMA's 22. That is the turning point — a clear benchmark for future work.

**Third — passive Reddit data can replace what surveys measure.** Surveys are expensive, slow, and late. My GLM picks up the **same exam-stress signal** from public Reddit posts — running all the time, at zero cost, with no recruitment. It picks out exam periods at +7.2% and breaks at −11.1%, both highly significant. Because exam dates are known months in advance, universities can **act before surges, not after**.

---

## Slide 13 — Conclusion (50 sec)

To conclude.

All three research questions are answered.

- **RQ1 — stress detection.** Hybrid VADER + RoBERTa, 70.6% agreement on 138,000 posts. Disagreements excluded openly.
- **RQ2 — calendar drivers.** Weekend drops 12–18%, exam rise 7.2%, break drop 11.1%. Stress follows the calendar, not a trend.
- **RQ3 — forecasting.** SARIMA wins overall at 21.61. Prophet wins once two years of data are available.

**Practical impact.** Universities can send counsellors **before** exam surges, not after. The system runs on **public data, at zero cost, with no surveys**. Use SARIMA today. Switch to Prophet as data grows.

**Future work.** Unfiltered control sample; LSTM with three or more years of data; cross-platform checks on X or Discord; a live warning dashboard; and multilingual subreddits.

Thank you. I am happy to take your questions.

---

## Timing check

| Slide | Title | Time | Weight |
|-------|-------|------|--------|
| 1 | Title | 0:25 | short |
| 2 | Background & RQ | 1:10 | **important** |
| 3 | Objectives | 1:00 | |
| 4 | Methodology 1–2 | 1:00 | |
| 5 | Methodology 3–5 | 1:00 | |
| 6 | Tools | 0:35 | short |
| 7 | NLP findings | 1:00 | |
| 8 | GLM findings | 1:30 | **most important** |
| 9 | Forecasting | 1:15 | **important** |
| 10 | Final forecast | 0:45 | |
| 11 | Discussion | 1:10 | **important** |
| 12 | Contributions | 1:15 | **important** |
| 13 | Conclusion | 0:50 | |
| **Total** | | **~12:00** | |

At a normal academic pace (~125 wpm), this lands at around 12 minutes. If you go faster (~140 wpm), about 11 minutes — leaves buffer for Q&A.

## Delivery tips

- **Pause** after each number — "Saturday is 18% lower" — let it land.
- **Memorise these numbers:** 138,058 · 70.6% · −18% Saturday · +7.2% exam · −11.1% break · SARIMA 21.61 · Prophet W4 9.8.
- **Slow down on Slide 8 (GLM)** — this is the evidence backbone.
- **Point at the chart on Slide 10** — read the shape, not the numbers.
- **Practice the opening and the closing "Thank you"** so nerves don't affect the first and last impression.
