# Assessment 3: Research Proposal

**Forecasting Mental Health Sentiment Surges: A Time-Series Analysis of Reddit Discourse**

---

## 1. Introduction

### 1.1 Background and Context

The rapid expansion of online academic communities has fundamentally transformed how university students communicate, seek support, and express emotional distress. Platforms such as Reddit host large, topic-specific communities (subreddits) where students openly discuss academic pressures, mental health challenges, and experiences related to university life. These digital spaces generate vast amounts of unstructured textual data that offer valuable opportunities for understanding student well-being at scale. Unlike traditional institutional surveys or clinical assessments, which capture only snapshots of mental health at specific moments, online communities provide continuous, real-time records of student sentiment and emotional expression.

Advances in data science, particularly in Natural Language Processing (NLP) and statistical time-series modelling, now enable the extraction of meaningful psychological signals from online discourse. This technological shift is gradually moving mental health monitoring from retrospective, survey-based approaches toward continuous, data-driven analysis. Rather than waiting for students to self-report distress through institutional channels, researchers can now analyze the linguistic patterns embedded in online conversations to detect emerging mental health trends. This represents a significant methodological opportunity, as social media data offers granularity, timeliness, and scale that traditional methods cannot achieve.

Student mental health has emerged as a critical global concern, particularly within higher education institutions. Academic workload, high-stakes examinations, financial pressures, and social isolation are consistently linked to elevated levels of anxiety, depression, and psychological distress. Traditional monitoring approaches, including self-report surveys, campus mental health services uptake, and institutional assessments suffer from well-documented limitations: low response rates, reporting bias, temporal delays, and inability to capture the full spectrum of student distress. Consequently, there is growing scholarly interest in leveraging digital traces from online platforms to provide real-time or near-real-time indicators of mental health trends, enabling universities to respond proactively rather than reactively.

### 1.2 Problem Statement

Despite the potential of social media data for monitoring student well-being, several critical challenges remain unresolved. First, most existing research focuses on static sentiment classification or cross-sectional analyses, with limited attention to the temporal dynamics of emotional expression. While researchers have successfully developed methods to classify individual posts as positive, negative, or neutral, they have rarely examined how aggregated sentiment signals fluctuate over time in response to predictable academic events such as midterms or final examinations. This temporal gap is particularly problematic because stress among students is not randomly distributed but episodic, clustering around specific academic calendar events.

Second, significant debate persists regarding the most effective methods for detecting mental health signals from social media text. Lexicon-based sentiment models (such as VADER) offer interpretability and computational efficiency but often fail to capture context, sarcasm, and implicit expressions of stress. In contrast, transformer-based deep learning models (such as RoBERTa and BERT) demonstrate superior accuracy in detecting nuanced emotional states but sacrifice interpretability and require greater computational resources. To date, few studies have systematically compared these approaches or explored hybrid methods that balance accuracy with explainability.

Third, there is limited integration of NLP-derived sentiment signals with rigorous statistical forecasting approaches. Existing research treats sentiment as isolated findings rather than as input for predictive models. When NLP and forecasting have been combined, the methodological rigor often diminishes, particularly regarding the handling of count-based data exhibiting overdispersion and temporal autocorrelation. This gap is consequential: without predictive integration, sentiment analysis remains descriptive rather than actionable, providing historical insights but no forward-looking capability for intervention planning.

### 1.3 Research Objective and Specific Objectives

This research proposal outlines a systematic framework that integrates NLP-based sentiment analysis with statistical time-series forecasting to examine temporal patterns in student emotional expression on social media and identify periods of heightened psychological demand. The overarching goal is to develop and validate a proactive, data-driven system capable of predicting stress surges among students without relying on institution-specific academic calendars, thereby enabling timely mental health interventions.

To achieve this primary objective, the research will pursue three specific sub-objectives:

**Objective 1:** Analyze temporal fluctuations and trends in aggregated student sentiment over extended periods, with particular focus on identifying recurring periods of elevated psychological strain. This involves extracting stress-related posts, aggregating them into time-indexed counts, and examining autocorrelation and seasonality patterns.

**Objective 2:** Compare the effectiveness of lexicon-based and transformer-based NLP models in detecting mental health-related signals from student-generated social media content. This comparative analysis will evaluate both the sensitivity and specificity of different approaches and explore whether hybrid methods can combine the strengths of both paradigms.

**Objective 3:** Apply appropriate statistical and time-series forecasting models to sentiment-derived count data to predict future periods of increased psychological stress. This involves fitting count regression models, evaluating temporal dependencies, and developing forecasts that can be operationalized for resource allocation and intervention planning.

### 1.4 Significance and Contributions

This research contributes to the field in three distinct dimensions. **Theoretically**, it bridges two traditionally separate research domains, text-based sentiment analysis and statistical forecasting, offering a novel methodological framework for understanding student mental health. By integrating NLP with count-based statistical modeling and time-series forecasting, the project advances both the data science and higher education fields.

**Practically**, the research has immediate institutional applicability. Universities struggle to allocate limited mental health resources efficiently. By providing predictive insights into periods of heightened stress, this framework enables administrators to proactively allocate counseling services, schedule support programs, and deploy interventions at moments of greatest need. This capability transforms reactive mental health responses into proactive, evidence-based planning.

**Ethically and socially**, this project contributes to broader Sustainable Development Goals, specifically SDG03 (Good Health and Well-Being) and SDG04 (Quality Education). By developing tools to monitor and support student mental health, the research promotes evidence-based strategies to safeguard well-being in academic settings. Furthermore, the research emphasizes responsible data use, privacy protection, and transparent algorithmic decision-making, aligning with contemporary ethical standards in AI and data science.

### 1.5 Scope of the Research Proposal

This proposal outlines a capstone research project focused on Reddit-based academic communities as a case study for predictive mental health monitoring. The research encompasses data collection, NLP implementation, statistical modeling, and forecasting, with explicit attention to methodological limitations and ethical considerations. While the project uses Reddit data from specific subreddits, findings may be generalizable to other online academic communities with similar characteristics. However, the proposal acknowledges that social media users may not represent the full student population, and results should be interpreted within this context.

The project does not involve direct intervention with students, clinical diagnosis, or institutional policy changes. Rather, it provides a methodological foundation and proof-of-concept that could inform future institutional applications. Furthermore, while the research addresses several methodological gaps identified in the literature, it recognizes that some challenges, such as the inherent noise in social media data and the difficulty of validating ground-truth stress, remain complex and may require ongoing refinement.

---

## 2. Literature Review

This literature review synthesizes research at the intersection of Natural Language Processing (NLP), statistical modeling, and time-series forecasting in the context of student mental health monitoring. Rather than presenting exhaustive coverage of each domain, this section identifies critical methodological gaps that the proposed research directly addresses.

### 2.1 NLP-Based Sentiment Analysis in Mental Health Research

#### 2.1.1 NLP for Mental Health Detection in Online Communities

Natural Language Processing has emerged as a pivotal tool for analyzing textual data to identify mental health indicators. Social media platforms including Reddit provide abundant, publicly accessible data reflecting users' emotional states and behavioral patterns. Unlike traditional institutional surveys limited by low response rates and reporting bias, NLP techniques enable quantification of stress and anxiety by analyzing linguistic cues embedded in user-generated content, offering near real-time insights into psychological trends. This capability is particularly valuable for student mental health monitoring, as Reddit communities provide continuous documentation of academic stress without relying on self-reporting mechanisms.

#### 2.1.2 Lexicon-Based vs. Transformer-Based Approaches

Two primary NLP paradigms currently dominate sentiment analysis research. Lexicon-based models such as VADER (Valence Aware Dictionary and Sentiment Reasoner) rely on predefined dictionaries associating words with sentiment scores. These approaches offer interpretability and computational efficiency, making them suitable for large-scale datasets. However, lexicon-based methods demonstrate critical limitations: they struggle with context, fail to recognize sarcasm, and miss implicit expressions of stress. For example, statements like "I can't even" may indicate high stress but appear neutral to lexicon-based systems.

In contrast, transformer-based deep learning models such as BERT and RoBERTa leverage contextual embeddings to detect complex emotional patterns, demonstrating superior accuracy in identifying nuanced psychological states common in academic discourse. These models excel at capturing implicit expressions and contextual nuance. However, they sacrifice interpretability and require substantially greater computational resources, creating practical deployment challenges.

#### 2.1.3 Temporal and Count-Based Limitations

Despite methodological advances, sentiment analysis in current literature remains fundamentally descriptive rather than predictive. Most studies report sentiment scores at single time points or averaged over extended periods, overlooking day-to-day or week-to-week variations critical for understanding episodic student stress. Furthermore, sentiment is typically treated as continuous scores rather than discrete counts appropriate for statistical forecasting and count regression models. This critical gap treating sentiment as a descriptive metric rather than as input for predictive frameworks, motivates the integrated approach proposed in this research. By converting sentiment classifications into time-indexed count data and embedding this in statistical forecasting pipelines, the research moves beyond documenting what happened to predicting what will happen.

### 2.2 Statistical Modeling of Count-Based Data

#### 2.2.1 Poisson and Negative Binomial Regression

After NLP classification, researchers commonly aggregate stress mentions into count data representing high-stress mentions per defined time period (day, week). Count data are discrete and non-negative, naturally suited to specialized regression approaches. Poisson regression assumes mean-variance equality, suitable for well-behaved datasets but unrealistic for social media signals, which tend to be sparse and irregular.

Negative Binomial regression accommodates overdispersion, where variance exceeds the mean, commonly observed in Reddit data. Specifically, Reddit stress mentions exhibit minimal activity on typical days but sharp spikes around predictable academic events such as midterms and final examinations. This pattern of sparsity with episodic bursts makes Negative Binomial regression particularly appropriate for modeling student mental health signals from social media.

#### 2.2.2 Temporal Autocorrelation in Count Data

Beyond static modeling, discrete time-series models capture temporal autocorrelation in count data. Poisson autoregressive (PAR) models extend standard Poisson frameworks by allowing expected counts at time t to depend on past observations, explicitly modeling temporal dependencies while preserving count properties. This is essential for social media data exhibiting clustering: elevated stress mentions at one time point increase the likelihood of elevated mentions at subsequent time points, reflecting genuine patterns in student behavior rather than random fluctuations.

These temporal approaches prove particularly relevant for understanding how academic stress propagates through online communities, with mentions clustering not randomly but in response to identifiable academic calendar events.

### 2.3 Time-Series Forecasting Approaches

#### 2.3.1 Classical Forecasting Models

Once count-based stress signals are derived, forecasting enables prediction of future stress surges, facilitating proactive resource allocation and interventions. Behavioral signals characteristically exhibit recurring patterns such as weekly cycles alongside spikes associated with academic deadlines. Classical forecasting models including ARIMA (AutoRegressive Integrated Moving Average) and SARIMA (Seasonal ARIMA) capture trends, autocorrelation, and seasonality, making them suitable for structured temporal data aligned with academic calendars. However, these methods assume linear relationships and struggle with sudden spikes in count data, often requiring extensive preprocessing and data transformation.

#### 2.3.2 Modern Forecasting Approaches

Modern approaches address classical limitations more effectively. Prophet, developed by Facebook, automatically models seasonality and incorporates holiday or event effects, proving particularly suited for academic-related stress forecasting where specific calendar events (midterms, finals, semester breaks) drive predictable patterns. LSTM (Long Short-Term Memory) networks capture long-term dependencies and non-linear temporal trends, though requiring larger training datasets and greater computational resources than traditional approaches. Studies applying these approaches to social media sentiment demonstrate feasibility of forecasting peaks in activity and negative sentiment, yet systematic integration with NLP-derived stress counts remains sparse in the literature.

### 2.4 Critical Research Gaps and Proposed Solutions

#### 2.4.1 Integration Gap

Despite substantial advances across NLP, statistical modeling, and forecasting domains, significant integration gaps persist. First, most research remains descriptive, documenting sentiment trends retrospectively without developing predictive frameworks capable of anticipating future stress surges. Second, integration across methodological domains is limited, few studies combine NLP-derived stress counts with robust statistical models and time-series forecasting in unified frameworks. Third, while ethical and representativeness concerns are acknowledged in literature, they are rarely operationalized in modeling practices.

#### 2.4.2 How This Research Addresses Gaps

This proposal explicitly addresses these integration gaps by: (1) developing a hybrid NLP approach combining lexicon-based interpretability with transformer-based accuracy; (2) converting sentiment classifications into count-based statistical models; (3) embedding count models within time-series forecasting pipelines; and (4) prioritizing ethical considerations including privacy protection, transparent decision-making, and acknowledgment of social media representation limitations. By systematically comparing NLP approaches and demonstrating their integration with forecasting, this research establishes a methodological bridge across traditionally separate fields, advancing both data science practice and higher education mental health monitoring capacity.

#### 2.4.3 Rationale for Integrated Methodological Approach

The significance of this integrated approach lies in its operational feasibility and practical utility. Previous research has demonstrated the promise of each component, NLP sentiment detection, count-based statistical modeling, and time-series forecasting in isolation. However, the absence of a unified framework has limited the translation of academic insights into actionable intelligence. By integrating these three methodological streams, this research creates a complete pipeline: extracting mental health signals from text to quantifying temporal patterns to forecasting future surges. This progression transforms sentiment analysis from a retrospective descriptive tool into a proactive early warning system. Furthermore, by employing a hybrid NLP approach and rigorous count-based statistical methods rather than relying on simplified sentiment scoring, the research maintains both scientific rigor and practical interpretability. The framework is designed to be replicable across different academic communities and potentially adaptable to other domains requiring continuous monitoring of population-level psychological indicators, such as employee well-being in organizational settings or patient sentiment in healthcare contexts.

---

## 3. Methodology

This section outlines the comprehensive methodological framework integrating NLP-based sentiment analysis, statistical count-based modeling, and time-series forecasting to predict mental health sentiment surges. The methodology is designed to achieve the three research objectives through a systematic pipeline: data collection, sentiment classification, count aggregation, statistical analysis, and forecasting. All procedures are guided by principles of methodological rigor, transparency, and ethical practice.

### 3.1 Research Design and Overall Framework

This research employs a sequential mixed-methods design combining qualitative text analysis (via NLP) with quantitative statistical modeling and forecasting. The design is structured as a three-stage pipeline: (1) extracting and classifying sentiment from Reddit posts; (2) aggregating sentiment classifications into time-indexed count data; and (3) applying statistical and time-series models to these counts for prediction and forecasting.

The research framework is quasi-experimental in nature, utilizing historical observational data rather than experimental manipulation. Data will be collected retrospectively from public Reddit discourse spanning an academic semester (approximately 16 weeks), encompassing multiple cycles of academic stress events (midterms and final examinations). This retrospective observational design allows examination of natural variation in stress patterns without requiring intervention or direct participant contact, minimizing ethical complications while maximizing ecological validity.

The analytical approach is fundamentally iterative and comparative. Rather than committing to a single method, the research systematically compares competing approaches at each analytical stage—lexicon-based vs. transformer-based NLP models, Poisson vs. Negative Binomial regression, and classical vs. modern time-series forecasting models—enabling evidence-based selection of optimal methods based on data fit and predictive accuracy.

### 3.2 Data Collection: Source, Scope, and Selection Criteria

#### 3.2.1 Data Source and Platform Justification

Data will be collected from Reddit, specifically from publicly accessible academic-oriented subreddits including r/college, r/students, r/mentalhealth, and discipline-specific communities (e.g., r/learnprogramming, r/chemistry). Reddit provides several advantages for this research: (1) large volume of authentic user-generated content; (2) public accessibility without authentication barriers; (3) temporal metadata enabling precise time-stamping of posts; (4) topical organization through subreddits facilitating targeted data collection; and (5) user anonymity reducing reporting bias compared to institutional surveys.

The decision to use Reddit specifically reflects the target population (predominantly undergraduate and graduate students) who actively engage with academic-focused subreddits. While Reddit users may not perfectly represent the full student population, they constitute a substantial and diverse community whose discourse reflects authentic academic stress patterns without the social desirability bias inherent in formal reporting mechanisms.

#### 3.2.2 Temporal Scope and Academic Calendar Alignment

Data collection will span one complete academic semester (16 weeks) covering typical academic calendars in the Northern Hemisphere (September to December or January to April, depending on institutional schedules). This temporal window is sufficient to capture multiple stress cycles: regular coursework weeks, midterm examination periods, final examination periods, and post-examination recovery weeks. The semester-long timeframe enables identification of recurring patterns while remaining manageable for manual validation of NLP classifications.

Within each day during the collection period, stress-related posts will be aggregated to create daily count data. This daily aggregation level balances temporal granularity (capturing day-to-day variation) with computational feasibility and sufficient sample sizes within each time unit.

#### 3.2.3 Inclusion and Exclusion Criteria

**Inclusion Criteria:**
- Posts written in English language
- Posts authored during the 16-week data collection period
- Posts containing explicit discussion of academic stress, anxiety, mental health challenges, or related emotional expressions
- Posts with complete metadata (timestamp, author, subreddit, text content)

**Exclusion Criteria:**
- Deleted or removed posts (inaccessible for analysis)
- Posts authored by automated bots or service accounts
- Posts that are purely informational without emotional content
- Posts from non-student communities or off-topic subreddits
- Posts with highly ambiguous stress content requiring subjective interpretation beyond NLP capability

The inclusion criteria prioritize relevance to student mental health and academic stress, while exclusion criteria ensure data quality and analytical validity. Posts will be initially filtered using keyword matching (e.g., "stress," "anxiety," "exam," "assignment," "depressed," "overwhelmed") to identify candidate stress-related posts, followed by manual verification sampling to validate inclusion criteria adherence.

### 3.3 Natural Language Processing: Hybrid Sentiment Analysis

#### 3.3.1 Hybrid NLP Approach Rationale

Rather than relying exclusively on either lexicon-based or transformer-based methods, this research employs a hybrid approach combining both paradigms. This design directly addresses the methodological gap identified in the literature review: balancing accuracy with interpretability.

The hybrid approach operates as follows: each post will be classified independently by both VADER (lexicon-based) and RoBERTa (transformer-based) models. Posts receiving concordant classifications (both models predicting the same sentiment direction) are classified with high confidence. Posts receiving discordant classifications are subjected to secondary analysis examining linguistic patterns and contextual factors to determine the most appropriate classification. This design preserves VADER's interpretability advantage—researchers can understand why specific linguistic features drove classifications—while leveraging RoBERTa's superior accuracy on implicit expressions and contextual nuance.

#### 3.3.2 VADER Implementation

VADER (Valence Aware Dictionary and Sentiment Reasoner) is a lexicon-based sentiment analysis tool specifically designed for social media text. VADER scores text on a 0-1 scale for negative, neutral, and positive sentiment, with compound scores ranging from -1 (most negative) to +1 (most positive).

Posts will be classified as stress-indicative if they: (1) contain VADER compound scores below -0.5, indicating strong negative sentiment; or (2) contain specific stress-related keywords not well-captured by raw sentiment scores (e.g., "exam," "deadline," "mental health crisis") combined with weakly negative sentiment (compound score between -0.5 and -0.1). This threshold-based approach acknowledges VADER's known limitation: statements like "I can't even study for this exam" may receive neutral sentiment scores despite clearly indicating stress.

#### 3.3.3 RoBERTa Implementation

RoBERTa (Robustly Optimized BERT Pretraining Approach) is a transformer-based deep learning model trained on large unlabeled text corpora, enabling capture of contextual semantic meaning beyond surface-level word associations. Pre-trained RoBERTa models fine-tuned on mental health and social media sentiment datasets will be employed. Specifically, models trained on Twitter sentiment data and mental health classification tasks have demonstrated strong performance on similar domains (Reddit).

RoBERTa will classify posts into binary categories: stress-indicative (confidence threshold > 0.7) and non-stress-indicative (confidence threshold < 0.7). The relatively high confidence threshold of 0.7 prioritizes precision over recall, minimizing false-positive stress classifications.

#### 3.3.4 Classification Reconciliation and Validation

For the ~15-20% of posts where VADER and RoBERTa classifications diverge, the principal investigator will manually review and classify each post using explicit decision rules: if a post contains stress-related keywords (e.g., "anxious," "depressed," "overwhelmed") or clearly expresses emotional distress, it is classified as stress-indicative regardless of sentiment scores.

To ensure consistency and reliability, the principal investigator will: (1) create a detailed coding manual with clear decision rules before beginning manual classifications; (2) recode a random sample of 200-300 posts (~10% of total) one week after initial coding to check for consistency (target: ≥85% agreement); and (3) spot-check classifications against original posts to identify any systematic errors. Classification accuracy, precision, recall, and F1-scores will be reported to validate the hybrid NLP approach.

### 3.4 Count Data Aggregation and Descriptive Analysis

#### 3.4.1 Temporal Aggregation

Following NLP classification, stress-indicative posts will be aggregated at the daily level, creating a time series of daily stress mention counts. Aggregation at the daily level balances temporal granularity with sufficient sample sizes and the natural rhythm of academic activities (classes, exams, assignment deadlines typically follow daily or weekly cycles).

For each day t within the 16-week observation period, a count variable Y_t will be computed representing the number of stress-indicative posts on that day, aggregated across all monitored subreddits. This creates a time series of 112 daily observations.

#### 3.4.2 Descriptive Statistics and Preliminary Analysis

Descriptive statistics will characterize the count distribution: mean, standard deviation, minimum, maximum, and percentiles. Temporal patterns will be visualized through time series plots showing raw counts over the 16-week period, enabling visual inspection of trends, seasonality, and potential anomalies.

Autocorrelation and partial autocorrelation functions (ACF/PACF) will be computed to examine temporal dependencies in count data, informing decisions about which time-series models are appropriate. Variance-to-mean ratios will be computed to assess overdispersion: if variance substantially exceeds the mean, Negative Binomial regression is preferred over Poisson regression.

Academic calendar markers (midterm dates, final examination periods, semester breaks) will be overlaid on temporal plots to visually examine associations between calendar events and stress surges, providing preliminary evidence for the hypothesis that academic events drive predictable stress patterns.

### 3.5 Statistical Modeling of Count Data

#### 3.5.1 Model Specification: Poisson and Negative Binomial Regression

Count data will be modeled using generalized linear models (GLMs) with count-specific probability distributions. Two competing models will be fitted:

**Model 1 - Poisson Regression:**
log(E[Y_t]) = β₀ + β₁(Days_to_Exam_t) + β₂(Days_Post_Exam_t) + β₃(Day_of_Week_t) + β₄(Week_Number_t)

**Model 2 - Negative Binomial Regression:**
log(E[Y_t]) = β₀ + β₁(Days_to_Exam_t) + β₂(Days_Post_Exam_t) + β₃(Day_of_Week_t) + β₄(Week_Number_t) + overdispersion parameter α

Where Y_t = count of stress-indicative posts on day t; Days_to_Exam_t = number of days until the next scheduled exam (0 if exam day, negative if post-exam); Days_Post_Exam_t = number of days since the most recent exam conclusion; Day_of_Week_t = categorical variable (dummy coded, Monday as reference); and Week_Number_t = academic week number (1-16).

The Negative Binomial model extends Poisson by allowing the variance-to-mean relationship to vary via the dispersion parameter α, accommodating overdispersion common in social media count data.

#### 3.5.2 Model Comparison and Selection

Both models will be fitted using maximum likelihood estimation. Model selection will be based on: (1) Akaike Information Criterion (AIC) with preference for models minimizing AIC; (2) Bayesian Information Criterion (BIC) with similar preference; (3) overdispersion assessment via deviance-to-df ratio; and (4) residual diagnostics examining whether residuals appear randomly distributed without systematic patterns.

The model with substantially lower AIC/BIC (Δ > 10) and superior residual diagnostics will be selected for feature interpretation and forecasting pipeline integration.

#### 3.5.3 Coefficient Interpretation and Feature Importance

Regression coefficients will be exponentiated to yield incidence rate ratios (IRRs): exp(β). These IRRs quantify the proportional change in expected stress counts for unit changes in predictors. For example, exp(β₁) = 1.25 indicates that each additional day closer to an exam is associated with 25% higher expected stress post counts. Confidence intervals will be computed for all parameters.

Statistical significance will be assessed using Wald tests with α = 0.05. Coefficients with p-values < 0.05 will be considered significant predictors and prioritized for integration into the forecasting model.

### 3.6 Time-Series Forecasting

#### 3.6.1 Prophet Framework Implementation

Following identification of significant predictors through count regression, time-series forecasting will employ the Prophet framework, developed by Facebook for business forecasting applications. Prophet decomposes time series into trend, seasonality, and holiday/event effects:

Y_t = Trend_t + Seasonality_t + Event_t + ε_t

Where Trend_t captures long-term direction (typically linear or piecewise linear in academic contexts); Seasonality_t captures recurring weekly and semester-level patterns; Event_t models the effects of known events (midterms, finals, semester start/end); and ε_t represents residual noise.

Exam dates and semester milestones identified during data exploration will be specified as events, allowing Prophet to quantify their effects on expected stress counts and adjust forecasts accordingly. This is particularly valuable because exam dates may shift slightly year-to-year; Prophet learns the magnitude of impact independently of specific calendar dates.

#### 3.6.2 Comparison with Classical (ARIMA) and Modern (LSTM) Approaches

For comparative validation, ARIMA (AutoRegressive Integrated Moving Average) and LSTM (Long Short-Term Memory) models will also be fitted. ARIMA is a classical approach using autoregressive, differencing, and moving average components, with order (p, d, q) determined via automated selection (auto.arima) or manual ACF/PACF analysis. ARIMA's assumption of linearity may limit performance given potential non-linear exam effects.

LSTM is a deep learning approach capturing non-linear temporal patterns. An LSTM network with 64 units, dropout regularization (rate = 0.2), and 2-layer architecture will be trained on 80% of data with 20% held for validation. Training will use Adam optimizer with early stopping to prevent overfitting.

All three models (Prophet, ARIMA, LSTM) will be evaluated using mean absolute error (MAE), root mean squared error (RMSE), and mean absolute percentage error (MAPE) on a hold-out test set (final 3 weeks of data, never seen during training). Model with superior test-set performance will be selected as primary forecasting approach; others will be retained for sensitivity analysis.

#### 3.6.3 Forecast Horizon and Confidence Intervals

Primary forecasts will generate predictions for 2-4 weeks ahead of the training data, providing sufficient lead time for universities to allocate mental health resources. All point predictions will include 80% and 95% confidence intervals estimated via bootstrap resampling, enabling probabilistic forecasts and risk quantification.

### 3.7 Limitations and Mitigation Strategies

#### 3.7.1 Representation Bias

**Limitation:** Reddit users may not represent the full student population. Reddit users tend to be younger, more digitally native, and possibly more comfortable discussing mental health online than general student populations.

**Mitigation:** (1) Results will be explicitly framed as findings from "Reddit-engaged students" rather than generalizing to all students; (2) comparisons with census data on student demographics will be provided; (3) sensitivity analysis will examine whether patterns differ across subreddits (discipline-specific vs. general), providing evidence of generalizability within Reddit communities.

#### 3.7.2 Data Quality and Measurement Error

**Limitation:** Automated NLP classifications may misclassify posts, particularly those containing sarcasm, humor, or subtle expressions of distress. Manual reconciliation is resource-intensive and potentially subject to coder bias.

**Mitigation:** (1) Hybrid NLP approach (VADER + RoBERTa) captures classification uncertainty and reduces single-method errors; (2) inter-rater reliability assessment ensures manual coding quality; (3) sensitivity analysis will repeat analyses using alternative classification thresholds to assess robustness.

#### 3.7.3 Temporal Autocorrelation and Forecasting Assumption Violations

**Limitation:** Count data exhibit temporal clustering (today's counts predict tomorrow's), potentially violating assumptions of standard regression. Stress surges cluster temporally rather than occurring randomly.

**Mitigation:** (1) Explicit incorporation of temporal autocorrelation through PAR models and autoregressive time-series approaches rather than treating data as independent; (2) residual diagnostics will test whether autocorrelation remains after modeling; (3) Durbin-Watson test and Ljung-Box test will formally assess residual independence.

#### 3.7.4 Limited Validation Against Ground Truth

**Limitation:** Without access to validated institutional mental health data, we cannot directly validate predictions against true student stress levels. Stress surges on Reddit may not perfectly correspond to institutional mental health demand.

**Mitigation:** (1) Forecasts will be framed as "Reddit-based stress indicators" rather than true clinical stress; (2) case studies will examine whether predicted stress surges correspond to known academic events (announced exam schedules, assignment deadlines); (3) recommendations for prospective validation using institutional counseling service utilization data will be provided.

### 3.8 Ethical Considerations

#### 3.8.1 Data Privacy and Anonymity

Reddit data are publicly available and do not require explicit informed consent; however, ethical principles demand careful stewardship. All data collection, processing, and analysis will adhere to Reddit's terms of service and research ethics guidelines. Specific safeguards include: (1) no direct identification of individual users; all analyses operate on aggregated counts, not individual-level data; (2) subreddit-level analysis rather than individual post-level reporting, protecting individual privacy; and (3) compliance with data protection regulations (GDPR where applicable) ensuring research data are stored securely with restricted access.

#### 3.8.2 Appropriate Use and Potential Harms

**Potential Harm:** Identifying predictable stress surges could enable inappropriate surveillance, targeting, or manipulation of vulnerable student populations by external actors with malicious intent.

**Safeguard:** Research findings will be disseminated to institutional stakeholders (student mental health services, university administrators) for constructive use in resource allocation and student support, with explicit guidance against surveillance applications. No raw data or identifying information will be publicly released.

#### 3.8.3 Algorithmic Transparency and Explainability

The hybrid NLP approach explicitly prioritizes explainability: VADER classifications are interpretable (specific linguistic features drive decisions), enabling stakeholders to understand why posts are classified as stress-indicative. This transparency is essential for institutional trust and appropriate application of findings.

#### 3.8.4 Acknowledgment of Limitations and Responsible Framing

Findings will be accompanied by clear discussion of limitations: the inability to diagnose clinical mental health conditions, the potential for Reddit-specific biases, and the preliminary nature of predictions. Recommendations will be framed as "supportive intelligence" for resource planning rather than definitive predictive instruments for clinical decision-making.
