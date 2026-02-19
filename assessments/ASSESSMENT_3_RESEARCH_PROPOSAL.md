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
