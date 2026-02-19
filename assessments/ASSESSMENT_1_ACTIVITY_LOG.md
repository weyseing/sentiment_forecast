# ASSESSMENT 1: ACTIVITY LOG WITH REFLECTIVE WRITING
**Master of Data Science - MRP5015 Capstone Project 1**

## REFLECTIVE WRITING: ACTIVITY LOG

- **Project Topic:** Forecasting Mental Health Sentiment Surges: A Time-Series Analysis of Reddit Discourse
- **Supervisor Meeting Dates:** 31 January 2026, 7 February 2026
- **Meeting Logs Attached:** Appendix A (Supervision Meeting Record 1 & 2)

---

## REFLECTIVE NARRATIVE

#### How My Literature Review Shaped My Project Direction

My initial proposal focused on simple VADER-based sentiment classification, but three key academic sources transformed my approach. VADER limitations for capturing sarcasm and implicit expressions (Hutto & Gilbert, 2014) were addressed through transformer-based contextual understanding (Wan & Dey, 2023). This led me to adopt a hybrid NLP approach comparing lexicon-based and deep learning methods.

Second, research on predicting depression via social media (De Choudhury et al., 2013) shifted my perspective from descriptive classification to predictive forecasting. Rather than asking "What sentiment do posts express?", I now ask "Can we predict when stress will surge?" This required integrating time-series forecasting—a methodological pivot that strengthened the project significantly.

Finally, count data regression frameworks (Cameron & Trivedi, 2013) revealed that aggregating sentiment into daily counts enables rigorous statistical modeling via Negative Binomial regression, accommodating the overdispersion inherent in social media data. This insight unified three typically separate domains: NLP, statistics, and forecasting.

#### Integration of Supervisor Guidance

During my first meeting (31 January 2026), my supervisor highlighted that classification alone lacks forecasting capability and practical utility. Her question—"How will you connect NLP outputs to statistical models?"—prompted deeper engagement with the Prophet framework (Taylor & Letham, 2018) and count-based approaches.

At our second meeting (7 February 2026), I presented the revised integrated proposal. She validated the approach but advised realistic scoping: Prophet (primary), ARIMA (comparative), LSTM (exploratory). She emphasized that "your NLP comparison is stronger than forecasting—that should be your showcase." She also recommended Poisson autoregressive models to explicitly capture temporal autocorrelation, ensuring theoretical grounding rather than ad-hoc methodology.

#### Challenges Encountered and Solutions

The methodological fragmentation between NLP and forecasting literatures initially seemed prohibitive—these communities rarely integrate. However, I reframed this gap as an innovation opportunity, positioning my work as a proof-of-concept bridging separate domains.

Technical complexity posed a second challenge. Rather than building RoBERTa from scratch (impractical within timeline), my supervisor guided me toward validated pre-trained models from Hugging Face, maintaining methodological rigor while respecting constraints.

Data validation became a strength: discordant VADER-RoBERTa classifications trigger manual review, creating a three-layer validation system that improves classification quality rather than reducing it.

#### Lessons Learned

Research design is iterative. I entered with a narrow sentiment-classification focus but learned that impactful research operates at domain intersections. Literature gaps represent innovation opportunities, not dead ends—the absence of integrated NLP-forecasting frameworks signalled unmet need, not impossibility.

My supervisor's question "Is your timeline feasible?" reframed constraints as strategic invitations. By prioritizing Prophet and treating LSTM as optional, I aligned ambition with realistic execution—a form of methodological rigor itself.

#### Technical Skills Gap Analysis

Through this literature review and project planning process, I have identified four critical skill gaps requiring mitigation before capstone execution in March 2026.

First, transformer-based deep learning implementation presents the most significant technical gap. While I possess intermediate Python proficiency and basic understanding of BERT architecture from coursework, I lack hands-on experience implementing RoBERTa models. To address this, I will complete Hugging Face's "NLP with Transformers" course (4-6 hours) focusing on loading pre-trained models and inference, study the BERT paper (Devlin et al., 2019) (2-3 hours), and conduct practice implementation on a sample of 100 Reddit posts (4-5 hours). This targeted approach prioritizes practical implementation over theoretical depth, recognizing that I need functional proficiency rather than innovation in model architecture. I aim to achieve this competency by end of February 2026.

Second, time-series forecasting represents a conceptual-to-practical transition gap. I understand forecasting principles from coursework but have no experience implementing Prophet, ARIMA, or LSTM models. My mitigation strategy involves completing Kaggle's "Time Series with Prophet" micro-course (3-4 hours), working through Facebook's official Prophet documentation (4-5 hours), implementing ARIMA on practice datasets to validate methodology (6-8 hours), and reviewing LSTM literature (Hochreiter & Schmidhuber, 1997) with Keras/PyTorch tutorials for hands-on coding (4-6 hours). This phased approach front-loads Prophet while maintaining flexibility for ARIMA and LSTM. I project completing this by early March 2026.

Third, count data regression requires moving from basic GLM understanding to specialized count-based approaches. I will review the foundational GLM text (McCullagh & Nelder, 1989), focusing on count data chapters (2-3 hours), work through Python code examples using statsmodels and scikit-learn (4-5 hours), study the "Negative Binomial Regression" textbook (Hilbe, 2011) for overdispersion and model selection (3-4 hours), and practice on publicly available datasets before applying to stress mentions (3-4 hours). Completing this by mid-February positions me to apply these models during statistical analysis in March. This progression from theory to application ensures both understanding and practical capability.

Fourth, manual classification and inter-rater reliability assessment demand systematic rigor I have not formally trained in. I will study the seminal work on reliability (Cohen, 1960) and Fleiss' extensions (2-3 hours), review coding manuals from published NLP validation studies (2-3 hours), develop my own detailed coding manual with explicit decision rules (3-4 hours), and conduct reliability checks on sample posts to ensure consistency (4-5 hours). This structured approach transforms manual review from a potential weakness into a methodological strength, aligning with supervisor feedback that "rigorous validation is rarely done this thoroughly." I aim to complete this by end of February.

Overall, my skills development timeline concentrates intensive learning in February 2026 (12-15 hours weekly) focused on RoBERTa implementation, count regression, and coding manual development. Early March 2026 adds LSTM and forecasting fine-tuning (10-12 hours weekly) while concurrent data processing begins, allowing task overlap that reduces critical path dependencies. From mid-March onward, my primary focus shifts to capstone project execution.

My readiness assessment is moderately confident with structured mitigation. The technical skills required are well-documented with accessible, high-quality learning resources. The primary challenge is time management—compressing significant learning into four to six weeks demands disciplined execution. However, my project's iterative nature enables overlap: VADER classification can proceed while learning RoBERTa; count aggregation can advance while mastering statistical models. By committing ten to fifteen hours weekly through February 2026, I am confident in achieving functional proficiency across all required tools before capstone execution begins.

---

## REFERENCES

Cameron, A. C., & Trivedi, P. K. (2013). *Regression analysis of count data* (2nd ed.). Cambridge University Press.

Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37–46. https://doi.org/10.1177/001316446002000104

De Choudhury, M., Gamon, M., Counts, S., & Horvitz, E. (2013). Predicting depression via social media. In *Proceedings of the International AAAI Conference on Web and Social Media*, 7(1), 128–138.

Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, 1, 4171–4186.

Hilbe, J. M. (2011). *Negative binomial regression* (2nd ed.). Cambridge University Press.

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

Hutto, C. J., & Gilbert, E. E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. In *Proceedings of the International AAAI Conference on Web and Social Media*, 8(1), 216–225.

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv preprint arXiv:1907.11692*.

McCullagh, P., & Nelder, J. A. (1989). *Generalized linear models* (2nd ed.). Chapman and Hall.

Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1), 37–45. https://doi.org/10.1080/00031305.2017.1380080

Wan, D., & Dey, N. (2023). Improving sentiment classification using a RoBERTa-based hybrid model. *Frontiers in Human Neuroscience*, 17, 1292010. https://doi.org/10.3389/fnhum.2023.1292010

---

# APPENDIX A: SUPERVISION MEETING RECORDS

## SUPERVISION MEETING RECORD - FIRST MEETING
(31 January 2026)

```
FACULTY OF ENGINEERING AND TECHNOLOGY
SCHOOL OF COMPUTING AND ARTIFICIAL INTELLIGENCE
MASTER IN DATA SCIENCE
MRP5015 CAPSTONE PROJECT 1: SUPERVISION MEETING RECORD
```

**Date:** 31 January 2026
**Time:** 9:00 PM - 10:00 PM
**Student:** [Your Full Name]
**Supervisor:** [Supervisor Name]

### 1. Updates from the Previous Meeting

Initial literature review on sentiment analysis in mental health contexts has been completed. A draft proposal outline with preliminary research questions was prepared, and initial exploration of Reddit data structure and accessibility was conducted.

### 2. Items Discussed in This Meeting

Student presented an initial proposal for a VADER-based sentiment classification system to identify mental health-related distress signals in Reddit student communities (r/college, r/students, r/mentalhealth), with descriptive analysis of sentiment trends. Supervisor identified three critical limitations requiring scope expansion: (1) classification alone is descriptive rather than predictive; (2) VADER classification is well-established in literature, limiting novelty; (3) lack of connection between NLP outputs and statistical models.

Following supervisor guidance, the student agreed to three major scope expansions: adopting a hybrid NLP approach comparing VADER (lexicon-based) with RoBERTa (transformer-based) to demonstrate methodological rigor; incorporating time-series forecasting (Prophet, ARIMA, LSTM) to enable predictive capability; and investigating count-based statistical modeling (Negative Binomial regression) rather than treating sentiment as continuous values. Supervisor emphasized that "impactful research bridges methodological domains. If you can show how NLP-derived signals feed into statistical forecasting, you have addressed a genuine gap in the literature."

Key literature reviewed during discussion: VADER sentiment analysis (Hutto & Gilbert, 2014); predicting depression via social media (De Choudhury et al., 2013); RoBERTa transformer models (Liu et al., 2019).

### 3. Work for the Coming Meeting

**Action Items (Due 7 February 2026):**

Student will conduct a comprehensive literature deep-dive into RoBERTa (Liu et al., 2019), Prophet forecasting (Taylor & Letham, 2018), count data regression (Cameron & Trivedi, 2013; Hilbe, 2011), and LSTM for time-series analysis (Hochreiter & Schmidhuber, 1997). The student will develop a hybrid NLP classification pipeline with explicit decision rules for reconciling discordant VADER-RoBERTa classifications, create implementation timelines for Prophet, ARIMA, and LSTM, and conduct a detailed feasibility assessment.

**Anticipated Challenges:**

Three primary challenges were identified: technical complexity of implementing RoBERTa and LSTM models; timeline pressure given the need to integrate multiple methodologies within 8 weeks; and resource requirements for manual data validation (approximately 15-20% of posts). Supervisor provided guidance to mitigate these challenges: "Start with VADER + RoBERTa comparison as your core contribution. Forecasting is important but secondary. Use the hybrid NLP approach to demonstrate methodological rigor. Do not attempt to innovate on all fronts simultaneously."

---

**Supervisor's Signature**
________________________________

**Student's Signature**
________________________________

---

## SUPERVISION MEETING RECORD - SECOND MEETING
(7 February 2026)

```
FACULTY OF ENGINEERING AND TECHNOLOGY
SCHOOL OF COMPUTING AND ARTIFICIAL INTELLIGENCE
MASTER IN DATA SCIENCE
MRP5015 CAPSTONE PROJECT 1: SUPERVISION MEETING RECORD
```

**Date:** 7 February 2026
**Time:** 2:00 PM - 3:00 PM
**Student:** [Your Full Name]
**Supervisor:** [Supervisor Name]

### 1. Updates from the Previous Meeting

All assigned tasks were completed ahead of schedule. Comprehensive literature reviews have been conducted on RoBERTa transformer models, Prophet forecasting methodology, LSTM networks, and count data regression approaches. A revised project proposal incorporating hybrid NLP and forecasting components has been drafted. A detailed project timeline with risk assessment was prepared, and critical technical skill gaps have been identified with corresponding mitigation strategies.

### 2. Items Discussed in This Meeting

Student presented a substantially revised proposal integrating four methodological components: hybrid NLP approach combining VADER (lexicon-based) and RoBERTa (transformer-based) with manual reconciliation of discordant classifications; statistical modeling comparing Poisson and Negative Binomial regression for count data; time-series forecasting including Prophet (primary method), ARIMA (comparative baseline), and LSTM (exploratory if time permits); and an integration framework showing the pipeline from NLP classifications through count aggregation to statistical modeling and forecasting.

Supervisor validation: "This is exactly what the field needs. Most sentiment analysis remains descriptive. Your contribution—demonstrating how NLP outputs become inputs to predictive systems—is novel and impactful." However, scope concerns were raised: "Eight weeks is tight for this scope. Can you realistically manage three forecasting methods?" Supervisor recommended explicit prioritization: Prophet is essential (automatic seasonality handling is ideal for academic calendar structure); ARIMA serves as a quick computational baseline; LSTM is optional and should be implemented only if Prophet and ARIMA are completed ahead of schedule.

Literature review informed all methodological choices. RoBERTa's superior ability to capture contextual nuance that VADER misses (Liu et al., 2019) justifies the transformer approach for implicit stress expressions. Prophet's automatic seasonality and event-effect modeling directly address academic calendar structure (Taylor & Letham, 2018). Count data literature demonstrates that treating sentiment as continuous is theoretically inappropriate for social media data exhibiting overdispersion (Cameron & Trivedi, 2013; Hilbe, 2011). LSTM literature indicates potential for non-linear temporal pattern capture but also reveals high computational and training data requirements (Hochreiter & Schmidhuber, 1997).

Additional methodological guidance: explore Poisson autoregressive (PAR) models to explicitly capture temporal autocorrelation; employ overdispersion assessment and AIC/BIC criteria for model selection; develop a detailed coding manual with explicit decision rules prior to manual validation; conduct inter-rater reliability assessment by recoding 10% of posts one week after initial coding, targeting ≥85% agreement.

### 3. Work for the Coming Meetings

**Priority Action Items (Due 28 February–3 March 2026):**

Student will complete RoBERTa implementation tutorials from Hugging Face with practice implementation on 100 sample posts by 28 February. Comprehensive study of count data regression with emphasis on overdispersion assessment and model selection procedures is due by 28 February. Development of comprehensive NLP classification decision rules and detailed coding manual is due by 21 February. Reddit data collection for the 16-week observation period will commence, with preliminary VADER classification of 1,000 sample posts and data quality assessment (deleted posts, bots, off-topic content) due by 28 February. Final proposal revision incorporating PAR models, count regression depth, and risk mitigation strategies is due by 3 March.

**Challenge Mitigation Strategies:**

For RoBERTa implementation complexity, supervisor guidance emphasized using pre-trained models from the Hugging Face library rather than training from scratch, noting that "your innovation is in the hybrid comparison and integration with forecasting, not in model architecture." For timeline pressure with multiple forecasting methods, the recommended strategy prioritizes Prophet as essential, ARIMA as a quick baseline comparison, and LSTM as optional only if preceding methods are completed ahead of schedule. For manual NLP validation effort (estimated 15-20% of data, approximately 30 hours), supervisor recommended front-loading this work in late February and early March, using breaks between other tasks for manual review, and maintaining a progress tracking sheet.

---

**Supervisor's Signature**
________________________________

**Student's Signature**
________________________________

---

**END OF ASSESSMENT 1**