# Presentation Script — CP2 Assessment 2
**Forecasting Mental Health Sentiment Surges**
Target: ~12 minutes · 13 slides · ~1 minute each

---

## Slide 1 — Title (30 sec)

Good morning, everyone. Thank you for being here.

My name is Heng Wey Seing, student ID 24042426. My supervisor is Prof. Dr Selina Low Yeh Ching.

Today I will present my Capstone Project 2, titled **"Forecasting Mental Health Sentiment Surges — A Time-Series Analysis of Reddit Discourse."**

In simple terms, this project tries to answer one question: **can we predict when university students will be most stressed, by reading what they post on Reddit?**

I will walk you through the problem, my method, the results, and what this project adds. Let us begin.

---

## Slide 2 — Background & Research Question (1 min 30 sec)

Let me start with the problem.

Research by Auerbach in 2018, and Lipson in 2022, shows that **one in three university students** has anxiety or depression. This is a very serious health problem.

But the way universities measure it today is slow. They depend on surveys — surveys are expensive, they run only once or twice a year, and they capture stress **after** it has already happened. By the time the data comes back, the exam is already over, and the student has already suffered.

Reddit is very different. Students post on Reddit **openly, anonymously, with timestamps, and in real time**. That means Reddit could be an **early warning signal** — if we can read it properly.

But when I read past studies, I found that no one has done the full job from start to finish. There are **three specific gaps**:

**First — the integration gap.** Most studies stop at labelling posts. They mark posts as stressed or not stressed, and that is it. Very few researchers go further and predict what will happen next.

**Second — the classification gap.** Studies use VADER **or** RoBERTa, but not both together. And when the two tools disagree, they usually just throw those posts away without saying so or explaining why.

**Third — the time-coverage gap.** Most datasets are shorter than one academic year, and use weekly averages. Daily patterns across two full years have not been studied.

So my research question is this: **Can a hybrid NLP and time-series pipeline, using Reddit posts, predict short-term stress surges accurately enough to help universities act *before* the surge, instead of after?**

---

## Slide 3 — Research Objectives (1 min 15 sec)

To answer that research question, I set three specific objectives — one for each gap I found in the literature.

**Objective 1 — for the classification gap.** Detect stress in Reddit posts using **two NLP models together**, VADER and RoBERTa. When both agree, the label is high confidence. When they disagree, the post is **transparently excluded** — originally I planned to review those manually, but with 40,500 disagreement cases this was not feasible, and the 97,527 agreement set was already large enough for robust modelling. This gives the output at Stages 1 and 2.

**Objective 2 — for the time-coverage gap.** Find out **what drives daily stress counts over a full two-year period**. I group the labelled posts into a 705-day time series and fit a Negative Binomial regression. This measures the effect of day of the week, exam periods, and semester breaks. Results are reported as **percentage change**, because that is easier for a non-statistician — for example, a university staff member — to understand.

**Objective 3 — for the integration gap.** Predict stress surges **21 days ahead**. I test Prophet and SARIMA with walk-forward cross-validation across four expanding windows, and pick the winner by mean absolute error.

Each objective gives a clear, measurable output at a specific stage of the pipeline, so progress can be checked from start to end.

---

## Slide 4 — Methodology, Stages 1 and 2 (1 min 15 sec)

These two stages handle data collection and labelling.

**Stage 1 — Collection.** I used the **Arctic Shift API**, which gives free access to Reddit's historical archive. No API key, no rate limits, no cost. I downloaded posts from four subreddits — **r/college, r/Students, r/GradSchool, and r/mentalhealth** — over two years, from January 2024 to December 2025. I used a list of 30 stress-related keywords to cast a wide net and catch as many relevant posts as possible. For r/mentalhealth, posts also had to match a student keyword like "exam", "semester", or "campus", so the data stays focused on academic stress. In total, **138,058 posts collected**.

**Stage 2 — Classification.** Each post is checked by **two models**, not one.

- **VADER** is a dictionary-based tool from Hutto and Gilbert in 2014. It is very fast, but it misses sarcasm and context.
- **RoBERTa** is a deep learning model from Liu and colleagues in 2019. It understands context, including sarcasm and negation, but it is slower.

The main idea of my method: I use **agreement between the two models as a confidence signal**. Both say stressed → label 1. Both say not stressed → label 0. They disagree → transparently excluded from the modelling set. I originally planned to review these disagreements manually, but with around 40,500 disagreement cases the volume was too large for one researcher, and the agreement set was already big enough to model robustly. What makes my approach different from prior work is not that I kept the disagreements, but that I **report them openly as a limitation** instead of silently dropping them.

---

## Slide 5 — Methodology, Stages 3, 4, and 5 (1 min 15 sec)

These three stages turn the labels into insight.

**Stage 3 — Aggregate.** I group the labelled posts by date to build a **daily time series of 705 days**. Days with zero total posts are dropped, because those are days where the API did not finish collecting — they are not real silent days. I also build **10 features** for modelling: day of the week, week number, exam period flag, semester break flag, post ratio, mean score, mean number of comments, and the share of posts from each subreddit.

**Stage 4 — Explain.** I fit a **Poisson GLM** and a **Negative Binomial GLM**, and compare them using AIC. Social-media counts are usually **overdispersed** — meaning the variance is far greater than the mean — and Negative Binomial handles this correctly. Hilbe showed this clearly in 2011. I report the results as **incidence rate ratios**, which can be read directly as percentage change.

**Stage 5 — Predict.** I test **Prophet and SARIMA** using walk-forward validation: four expanding windows, each forecasting 21 days ahead, always tested on data the model has never seen. Both models receive the same academic calendar events — Prophet through its holidays table, SARIMA through extra input variables — so the comparison is fair.

Why separate the GLM and the forecast? Because they answer different questions. **The GLM tells us *why* the counts shift.** **The forecast tells us *when* the next surge will hit.** Universities need both.

---

## Slide 6 — Tools and Justification (45 sec)

A quick word on the tools.

- **Arctic Shift API** — free, no API key, historical archive. I chose it over the official Reddit API because it has no cost and no rate limits.
- **VADER + RoBERTa** — hybrid, as explained earlier.
- **statsmodels** in Python for the GLM.
- **Facebook Prophet** for seasonality and holiday effects.
- **SARIMA** from statsmodels for the weekly pattern.

Every tool is open source, free, and easy to repeat — so any university could run this pipeline on their own.

---

## Slide 7 — Findings: NLP Classification (1 min 15 sec)

Let us look at what the NLP stage produced.

Out of **138,058 posts**, the two models **agreed on 70.6%** of them. That is **97,527 high-confidence labels**. Of those agreement cases, **51.3% were labelled as stressed**, and **19.4% as not stressed**.

The remaining **29.4% — about 40,500 posts** — were disagreements between VADER and RoBERTa. That 29.4% is roughly the share of unclear, sarcastic, or context-heavy posts where a simple dictionary tool and a deep learning model see things differently.

Originally, I planned to review these disagreements manually. But with 40,500 cases, the volume was too large for one researcher to get through. Since the 97,527 agreement cases were already a large, clean dataset for modelling, I excluded the disagreements from the analysis.

This is important for two reasons.

**First — transparency.** Past studies usually throw away disagreements without saying so, and the reader never sees that the choice was made. My pipeline reports this exclusion openly and quantifies exactly how many posts were dropped and why.

**Second — quality.** By keeping only the cases where two independent models agree, the labels that go forward into Stage 3 are more trustworthy than a single-model pipeline would produce.

So Stage 2 gives a clean, honest, and trustworthy labelled dataset, ready for count modelling.

---

## Slide 8 — Findings: GLM Results (1 min 45 sec)

This is the most important slide, because it answers the question: **what actually drives daily stress counts?**

First, which model wins. The Negative Binomial GLM beats Poisson clearly. The AIC gap is **550 points**, and the overdispersion ratio is **7.33** — meaning the variance is more than seven times the mean. Poisson cannot handle that, so Negative Binomial is the correct choice.

Now the results. I will report them as **incidence rate ratios**, or IRRs, which can be read directly as percentage change. Monday is the reference day.

**Day of the week effects:**
- **Saturday — 18% fewer stress posts** than Monday.
- Sunday — 12.7% fewer.
- Friday — 12.3% fewer.
- All three are highly significant, p less than 0.001.
- Monday through Thursday — no significant difference between each other.

What this means: students **switch off from school topics on weekends**. They are probably still stressed on Friday night, but they stop posting about school.

**Academic calendar effects:**
- **Exam periods — posts go up by 7.2%.** Significant.
- **Semester breaks — posts drop by 11.1%.** Significant.

**Subreddit mix effect:** when the share of r/mentalhealth posts goes up, academic stress posts drop by 44.5%. This makes sense — r/mentalhealth covers more than just school stress.

And the most important finding is a **negative one**: week number is **not significant**, p equals 0.81. Over two years, there is **no upward trend in stress posts**. Stress is cyclical — it follows the academic calendar — it is **not** getting worse year by year.

This matches stress-and-coping theory exactly.

---

## Slide 9 — Findings: Forecasting (1 min 30 sec)

Now the forecasting stage. This answers **when** the next stress surge will happen.

I compared two models — **Prophet and SARIMA** — using a method called **walk-forward cross-validation**. The idea is simple. I start with a smaller training window, forecast the next 21 days, then expand the window and forecast the next 21 days, and so on. Four windows in total. The models are always scored on data they have never seen before.

**Overall winner: SARIMA.** Mean MAE of **21.61**, compared to Prophet's **35.67**. On average, SARIMA's forecast is off by about 22 posts per day, which is quite good given the series averages 100 per day.

But the per-window detail tells a more interesting story:
- **Window 1 — SARIMA 28.4, Prophet 96.4.** With less than a year of data, Prophet struggles badly because it cannot see a full yearly cycle yet.
- **Window 2** — SARIMA 21.9, Prophet 18.2. Prophet catches up.
- **Window 3** — SARIMA 14.2, Prophet 18.4. Both are strong.
- **Window 4** — SARIMA 22.0, **Prophet 9.8**. Prophet actually **beats** SARIMA by a lot.

The lesson is clear. **SARIMA is the safer choice when training data is short**, because it models the 7-day weekly pattern directly. **Prophet becomes the better model once it has seen a full year cycle** — because only then can it learn yearly seasonality.

So the model choice depends on **how much historical data you have** — that is a simple rule for anyone deploying this.

---

## Slide 10 — Findings: Final Forecast Chart (1 min)

This chart shows the full two-year series, with **green shading for semester breaks** and **red shading for exam periods**, plus the **21-day forward forecast** on the right end.

You can see with your own eyes what the GLM confirmed with numbers. Counts **drop during breaks**, **rise during exams**, and **dip every weekend**. The forecast tail also includes a **95% confidence band**, so the audience — a university student-affairs team, for example — knows not just the best guess number, but also how uncertain that guess is. When the band is narrow, the model is confident; when it is wider, be careful.

The Prophet model was trained here on the **full dataset** with academic calendar events from the US, UK, and Australia — because the four subreddits come from different countries.

This is basically the **dashboard view** that a student-affairs or counselling team would check every week to plan ahead.

---

## Slide 11 — Discussion (1 min 30 sec)

So what do these findings actually tell us?

**First — the calendar effects.** Students post less on weekends, and more during exam periods. This matches existing stress-and-coping literature. What is new is the **daily level** and the **clear percentage numbers** — a Saturday drop of 18%, an exam-period rise of 7.2%. Past work only estimated these effects at the weekly level, and only reported direction, not size.

**Second — the model-choice finding.** SARIMA wins in the short run, Prophet wins in the long run. This is a simple rule: if you are deploying tomorrow with one year of data, use SARIMA. If you have two or more years, switch to Prophet. Most past work never tested this turning point because their datasets were too short.

**Third — the no-trend finding.** Over two years, stress volume did **not** grow. This is an important message for policymakers: the problem is **cyclical**, not getting worse. The right response is **timed support during exams**, not hiring permanent extra staff.

Now the **limitations** — three things I want to be honest about.

**Limitation 1:** The keyword filter pushes collection toward stressed content. This was on purpose — cast a wide net first, then Stage 2 confirms which posts are really stressed.

**Limitation 2:** We left out the 29.4% disagreement posts. Manual review was planned but not feasible at 40,500 cases for a single researcher. This gives clean labels, but we lose 40,500 posts that might still carry useful signal.

**Limitation 3:** Reddit users are not a random sample of students. The results reflect **what Reddit users say**, not the stress level of all students in general.

---

## Slide 12 — Contributions to Knowledge (1 min 30 sec)

Three contributions that go beyond what past studies have established.

**First — the first full, end-to-end pipeline.** This is the integration contribution. Past work either stops at text classification, or starts at forecasting with already-labelled data. No one has joined all five stages — scraping, hybrid NLP, aggregation, count regression, and walk-forward forecasting — in one single system that others can repeat. My pipeline closes that loop on 138,058 posts, with every script open source.

**Second — I give a concrete number for how much data Prophet needs.** Taylor and Letham, when they introduced Prophet in 2018, wrote that the model needs "sufficient" data to learn yearly seasonality, but they never gave a real number. My walk-forward results give one. At roughly two years of daily data, Prophet's Window 4 MAE drops to **9.8**, beating SARIMA's 22. That is the turning point — a clear benchmark that future researchers and practitioners can use.

**Third — I show that passive Reddit data can replace what surveys measure — automatically.** Surveys prove exam stress exists, but they are expensive, slow, one-time, and late. My GLM picks up the **same exam-stress signal** from public Reddit posts — running all the time, at zero cost, with no recruitment and no ethics paperwork. It adjusts for day of the week, subreddit mix, and post volume at the same time, then picks out exam periods at +7.2% and semester breaks at −11.1%, both highly significant. Because exam dates are published months ahead, universities can **act before surges happen**, not after. This builds on Lazer and others from 2009 — who argued that digital footprints can act as a cheap replacement for surveys — by giving a clear daily-level effect size.

---

## Slide 13 — Conclusion (1 min)

To conclude.

All three research questions are answered.

- **RQ1 — stress detection.** The hybrid VADER + RoBERTa pipeline worked, with 70.6% agreement on 138,000 posts. Disagreements are transparently excluded, not silently guessed.
- **RQ2 — calendar drivers.** Weekend drops of 12 to 18%, exam-period rise of 7.2%, semester-break drop of 11.1%. Stress follows the academic calendar, not a long-term trend.
- **RQ3 — forecasting.** SARIMA wins overall with MAE 21.61; Prophet becomes the better long-run choice once two or more years of daily data are available.

**Practical impact.** Universities can send counsellors **before** exam surges hit, not after. The system runs on **public data**, at **zero cost**, with **no surveys and no recruitment**. SARIMA is ready to use today; switch to Prophet as more data builds up.

**Future work.** An unfiltered control sample; LSTM models once we have three or more years of data; cross-platform checks on X or Discord; a live warning system feeding counselling dashboards; and multilingual subreddits for international universities.

Thank you for listening. I am happy to take your questions.

---

## Timing check

| Slide | Title | Time | Weight |
|-------|-------|------|--------|
| 1 | Title | 0:30 | short |
| 2 | Background & RQ | 1:30 | **important** |
| 3 | Objectives | 1:15 | |
| 4 | Methodology 1–2 | 1:15 | |
| 5 | Methodology 3–5 | 1:15 | |
| 6 | Tools | 0:45 | short |
| 7 | NLP findings | 1:15 | |
| 8 | GLM findings | 1:45 | **most important** |
| 9 | Forecasting | 1:30 | **important** |
| 10 | Final forecast | 1:00 | |
| 11 | Discussion | 1:30 | **important** |
| 12 | Contributions | 1:30 | **important** |
| 13 | Conclusion | 1:00 | |
| **Total** | | **~15:00** | |

Buffer built in: at a faster pace (~140 wpm) you will land around 13:00. At a slower, clearer academic pace (~115 wpm) you will land around 15:00. If you need to cut, start by trimming slides 6, 10, and 13 — the short ones are the easiest to shorten without losing evidence.

## Delivery tips

- **Pause** after each number — e.g. "Saturday is 18% lower" — let it land.
- **Memorise these numbers:** 138,058 · 70.6% · −18% Saturday · +7.2% exam · −11.1% break · SARIMA 21.61 · Prophet W4 9.8.
- **Slow down on Slide 8 (GLM)** — this is the evidence backbone.
- **Point at the chart on Slide 10** — don't read numbers, read the shape.
- **Practice the opening 30 seconds and the closing "Thank you"** so nerves don't affect the first and last impression.
