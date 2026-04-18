# CP2 Supervision Meeting Records
**MRP5025 Capstone Project 2 — Master of Data Science**
**Faculty of Engineering and Technology, School of Computing and Artificial Intelligence**

---

## Meeting Record 1

**Date:** 14 March 2026  
**Time:** 10:00 AM – 11:00 AM  
**Student:** Heng Wey Seing (24042426)  
**Supervisor:** Prof. Dr Selina Low Yeh Ching  

### Updates from the Previous Meeting

Completed CP1 (MRP5015) deliverables: literature review, research proposal, and activity log. The proposal outlined a five-stage pipeline integrating hybrid NLP classification (VADER + RoBERTa), Negative Binomial GLM, and comparative time-series forecasting (Prophet vs SARIMA). Reddit data collection strategy using the Arctic Shift API was finalised, covering four subreddits (r/college, r/Students, r/GradSchool, r/mentalhealth) over a two-year window (January 2024 to December 2025).

### Items Discussed in This Meeting

Student presented the CP1 research proposal as the baseline for CP2 implementation. Supervisor confirmed the overall pipeline design and provided the following guidance:

1. Proceed with the two-year (2024–2025) dataset rather than the one-semester subset to ensure sufficient observations for walk-forward forecasting validation.
2. Prioritise SARIMA over plain ARIMA to capture the weekly seasonal pattern; LSTM is deprioritised given the dataset size.
3. Ensure all five pipeline stages produce structured CSV outputs and reproducible Python scripts before beginning the CP2 written report.
4. Academic calendar covariates should cover multiple regions (US, UK, Australia) to reflect the international composition of the subreddits.

Supervisor noted: *"Your proposal is detailed enough to serve as the methodology chapter skeleton — map each pipeline stage directly to a section in Chapter 3."*

### Work for the Next Meeting

1. Execute `src/1_scrape_reddit.py` to collect the full two-year dataset.
2. Run `src/2_classify_sentiment.py` — hybrid VADER + RoBERTa classification.
3. Run `src/3_aggregate_counts.py` to produce the 705-day daily count series.
4. Fit Poisson and Negative Binomial GLMs (`src/4_model_glm.py`) and confirm overdispersion.
5. Run walk-forward forecasting (`src/5_forecast.py`) comparing Prophet and SARIMA.
6. Prepare `step_report.html` summarising results from all five steps for review at next meeting.

---

**Supervisor's Signature:** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Student's Signature:**

---

## Meeting Record 2

**Date:** 3 April 2026  
**Time:** Phone Discussion  
**Student:** Heng Wey Seing (24042426)  
**Supervisor:** Prof. Dr Selina Low Yeh Ching  

### Updates from the Previous Meeting

All five pipeline stages completed and outputs verified:

- **Step 1:** 138,058 records collected (68,217 posts, 69,841 comments) across four subreddits.
- **Step 2:** Hybrid classification yielded 70,788 stressed (51.3%), 26,739 not stressed (19.4%), 40,531 needs review (29.4%). Concordance rate: 70.6%.
- **Step 3:** 705-day daily time series produced; mean = 100.4, max = 199, min = 33 stressed posts/day.
- **Step 4:** Negative Binomial GLM fitted; overdispersion ratio = 7.33, ΔAIC vs Poisson = 550.26.
- **Step 5:** SARIMA(1,1,1)(1,1,1,7) mean MAE = 21.61; Prophet mean MAE = 35.67 overall.

`step_report.html` prepared covering all five steps with tables and charts.

### Items Discussed in This Meeting

Supervisor reviewed `step_report.html` and the pipeline scripts and dataset outputs via phone call. Key points discussed:

1. **GLM results:** Weekend effect findings are statistically meaningful — Saturday (−18.0%), Sunday (−12.7%), Friday (−12.3%) relative to Monday (all p < 0.001). Exam period (+7.2%) and semester break (−11.1%) are consistent with the literature.
2. **Forecasting:** SARIMA is the overall winner (MAE 21.61 vs 35.67). Prophet's strong Window 4 result (MAE 9.8 vs 22.0) when trained on the full dataset should be highlighted in the discussion as evidence that Prophet becomes competitive with sufficient data.
3. **NLP classification:** Supervisor queried the 29.4% needs-review exclusion rate. Student explained the conservative design choice. Supervisor agreed this is defensible and should be explicitly justified in the methodology chapter.
4. **Data quality:** The 705-day series in `3_daily_counts.csv` is continuous with no structural gaps after dropping API cut-off artefact days. No changes to scripts required.
5. **Scope:** LSTM deprioritised — confirmed by supervisor as appropriate given dataset size.

### Work for the Next Meeting

1. Begin writing the CP2 report — prioritise Chapters 1, 3, and 4 first.
2. Incorporate all figures (`4_residuals.png`, `5_cv_plot.png`, `5_final_forecast.png`) with numbered captions.
3. Write a dedicated discussion of why Prophet underperforms on shorter training windows but becomes competitive with full two-year data — link to Taylor & Letham (2018).
4. Submit draft report for supervisor review by 10 April 2026.

---

**Supervisor's Signature:** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Student's Signature:**

---

## Meeting Record 3

**Date:** 16 April 2026  
**Time:** Email Discussion  
**Student:** Heng Wey Seing (24042426)  
**Supervisor:** Prof. Dr Selina Low Yeh Ching  

### Updates from the Previous Meeting

CP2 report draft (`CP2_REPORT.md`) completed covering all five chapters: Introduction, Literature Review, Methodology, Findings & Results, and Conclusion. APA 7th edition citations applied, figures embedded with captions, and the draft submitted to supervisor via email on 10 April 2026.

### Items Discussed in This Meeting

Supervisor returned the annotated draft via email with five methodological queries. Each was addressed through a targeted revision to the corresponding section of the report:

1. **Stage 1 keyword filter:** Supervisor queried whether the 30-keyword filter would skew the dataset toward stressed posts. Student reframed Stages 1 and 2 as a screen-then-confirm cascade — Stage 1 as high-recall retrieval, Stage 2 as high-precision confirmation — and re-characterised the 51.3% concordant rate as confirmation-of-candidates rather than prevalence. Clarifications added to §3.3, §4.2; new limitation and unfiltered-control-sample follow-up added to §5.4 and §5.5.
2. **Day-of-week normalisation:** Supervisor asked whether Table 4.5 should account for mean total posts to distinguish a volume from a per-post effect. Recomputed table now reports mean total, mean stressed, and stress rate. Finding: volume swings 24.5% across the week while the rate is flat at 51.0%–51.6%. Narrative in §4.3.2 rewritten and GLM IRRs reinterpreted as volume effects in §4.4.2.
3. **IRR rationale:** Supervisor requested justification for reporting IRRs rather than raw regression coefficients. Rationale paragraph added to §3.5 and shorter forward-reference added to §4.4.2, covering three reasons: log-link uninterpretability, stakeholder percentage-change language, and cross-predictor comparability on a unit-free scale.
4. **Secular trend clarification:** Supervisor asked for a plain-language definition of *secular trend* in §4.4.2. In-text definition added and the sub-heading relabelled "Secular (long-term) trend"; the finding (IRR = 1.000, p = 0.811) is unchanged.
5. **Citation and reference audit:** Full cross-check performed — 48 in-text citations, 50 references, all consistent. One duplicate `## References` heading fixed. Three conference-proceeding entries (Baumgartner et al., 2020; Hutto & Gilbert, 2014; Vaswani et al., 2017) lack DOIs; optional stable URLs flagged for supervisor's decision.

Revised draft returned to supervisor with tracked changes for confirmation before final submission.

### Remaining Tasks Before Final Submission

This is the final supervision meeting for CP2. Supervisor confirmed the five revisions above are satisfactorily addressed and approved the report for submission subject to the following pre-submission housekeeping:

1. Resolve remaining TODOs at the top of `CP2_REPORT.md`: file-name formatting (`MDS_24042426_Heng Wey Seing_MRP5025_Selina Low_Assessment 1`), Assessment + Capstone cover pages from XLearn, auto-generated Table of Contents / List of Figures / List of Tables in Word.
2. Align the collection-window end date (19 December 2025) across §3.2, §3.4, §4.1, and §4.5.2.
3. Finalise figure and table numbering; verify all cross-references after conversion to Word / PDF.
4. Attach the CP2 Assessment cover page and submit the final report via the XLearn portal before the Capstone deadline.

---

**Supervisor's Signature:** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Student's Signature:**
