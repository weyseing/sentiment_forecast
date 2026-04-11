# Assessment 2: Literature Review

## Cover Page Information

**Course Code:** MRP5015 CAPSTONE PROJECT 1
**Activity:** Literature Review
**Project Title:** Forecasting Mental Health Sentiment Surges: A Time-Series Analysis of Reddit Discourse

**Student Name:** Heng Wey Seing
**Student ID:** 24042426
**Programme:** Master of Data Science (481MDS-ODL)
**Supervisor:** Prof. Dr Selina Low Yeh Ching
**Submission Date:** 7 Feb 2026

**Department:** Data Science and Artificial Intelligence
**School:** School of Computing and Artificial Intelligence
**Faculty:** Engineering and Technology
**Institution:** Sunway University

---

## Table of Contents

| No. | Content | Page |
|-----|---------|------|
| 1 | Introduction | 4 |
| 2 | Literature Review | 7 |
| 3 | Conclusion | 18 |
| 4 | References | 21 |

---

## 1. Introduction

### 1.1 Background and Context

The rapid expansion of online academic communities has transformed how university students communicate (Ks, 2024), seek support, and express emotional distress. Platforms such as Reddit host large, topic-specific communities (subreddits) where students openly discuss academic pressures, mental health challenges, and experiences related to university life. These platforms generate vast amounts of unstructured textual data that offer valuable opportunities for understanding student well-being at scale (Oryngozha et al., 2024). Advances in data science, particularly in Natural Language Processing (NLP) and statistical time-series modelling, now enable the extraction of meaningful psychological signals from online discourse, shifting mental health monitoring from retrospective surveys toward continuous, data-driven analysis (Saha et al., 2022).

Student mental health has emerged as a critical global concern, particularly within higher education. Academic workload, examinations, financial pressures, and social isolation are consistently linked to elevated levels of anxiety and psychological distress (Osman, 2025). Traditional monitoring approaches, such as self-report surveys and institutional assessments, are often limited by low response rates, reporting bias, and delayed insights. Consequently, there is growing scholarly interest in leveraging digital traces from online platforms to provide real-time or near-real-time indicators of mental health trends (Madrid‐Cagigal et al., 2025). This evolution raises methodological and ethical questions regarding the reliability, validity, and interpretability of sentiment signals derived from online text (Chhabra et al., 2025).

### 1.2 Problem Definition

Despite the potential of social media data for monitoring student well-being, several challenges remain. Most existing studies focus on static sentiment classification or cross-sectional analyses, with limited attention to the temporal dynamics of emotional expression (MacAvaney et al., 2018). Specifically, few studies examine how aggregated sentiment signals fluctuate over time in response to academic events, such as midterms or final examinations (Rodríguez-Ibánez et al., 2023). Additionally, debates persist regarding the most effective methods for detecting mental health signals, particularly when comparing lexicon-based sentiment models (e.g., VADER) and transformer-based models (e.g., RoBERTa). There is also limited integration of NLP-derived signals with rigorous statistical forecasting approaches that account for the count-based, over dispersed nature of sentiment data (Saeed & Cha, 2025). These gaps highlight the need for a systematic framework that combines sentiment extraction with predictive modelling to identify periods of high psychological demand among students.

### 1.3 Research Objective

This study aims to develop a systematic framework that integrates NLP-based sentiment analysis with statistical forecasting to examine temporal patterns in student emotional expression on social media and identify periods of heightened psychological demand, without relying on institution-specific academic calendars.

#### Specific Objectives

- To analyze temporal fluctuations and trends in aggregated student sentiment over time, with a focus on identifying recurring periods of elevated psychological strain.
- To compare the effectiveness of lexicon-based and transformer-based NLP models in detecting mental health-related signals from student-generated social media content.
- To apply appropriate statistical and time-series forecasting models to sentiment and count data to predict future periods of increased psychological stress.

### 1.4 Significance of the Study

The integration of NLP and statistical forecasting has the potential to transform how universities monitor and respond to student mental health concerns. By providing predictive insights into periods of heightened stress, this research can support administrators in proactively allocating mental health resources, scheduling interventions, and tailoring support programs to periods of greatest need. Moreover, the project contributes to Sustainable Development Goals, including health (SDG03) and education (SDG04), by promoting evidence-based strategies to safeguard student well-being. Beyond practical applications, the study also advances methodological innovation by bridging two traditionally separate fields—text-based sentiment analysis and count-based statistical forecasting—offering a novel approach to predictive mental health monitoring in online communities.

### 1.5 Scope of the Literature Review

This literature review critically examines research at the intersection of NLP, statistical modelling, and forecasting in the context of mental health monitoring within online academic communities. The review focuses on three interconnected domains: (1) NLP-based sentiment analysis for detecting psychological states in social media text, (2) statistical models for count data, including methods to handle sparsity and overdispersion, and (3) time-series forecasting techniques applied to behavioral and psychological signals. Studies that rely solely on non-academic sources, lack methodological transparency, or do not directly address temporal patterns in sentiment data are excluded to maintain analytical rigor. While the review emphasizes recent studies published primarily within the last decade, seminal foundational works are also considered to provide a comprehensive understanding of theoretical and methodological developments.

### 1.6 Organization of the Review

The remainder of this literature review is organized as follows. Section 2 reviews the theoretical and methodological foundations of sentiment analysis in mental health research, including both lexicon-based and transformer-based approaches. Section 3 examines statistical models for count data, highlighting techniques for handling sparsity, overdispersion, and temporal dependence. Section 4 explores time-series forecasting methodologies, including ARIMA and Prophet, and their applications in social and behavioral data contexts. Section 5 synthesizes findings across these domains, highlighting key limitations, unresolved debates, and methodological gaps in current research. Finally, the conclusion summarizes the major insights from the literature and identifies opportunities for future research, providing a clear rationale for the proposed capstone project titled Forecasting Mental Health Sentiment Surges: A Time-Series Analysis of Reddit Discourse. By integrating NLP and statistical forecasting, the project aims to provide a proactive, data-driven framework for monitoring student well-being, aligning with Sustainable Development Goals on health (SDG03) and education (SDG04).

---

## 2. Literature Review

### 2.1 NLP-Based Sentiment Analysis in Mental Health Monitoring

#### 2.1.1 NLP for Mental Health Detection in Online Communities

Natural Language Processing (NLP) has become a pivotal tool for analyzing textual data to identify indicators of mental health. Social media platforms, including Reddit and Twitter, provide abundant, publicly accessible data that reflect users' emotional states and behavioral patterns (Ks, 2024). NLP techniques enable the quantification of stress, anxiety, and depressive tendencies by analyzing linguistic cues in user-generated content. Compared to traditional surveys, which may suffer from self-report bias and limited temporal coverage, NLP offers near real-time insights into psychological trends (Villanueva-Miranda et al., 2025). Prior research has demonstrated the potential of NLP to detect mental health signals across populations, including university students, who often express academic stress and anxiety through online forums. By transforming unstructured text into measurable sentiment scores (Yang & Li, 2025), NLP forms the foundation for predictive frameworks that anticipate periods of elevated stress in online communities.

#### 2.1.2 Lexicon-Based Sentiment Analysis: Strengths and Limitations

Lexicon-based approaches are among the earliest methods employed in sentiment analysis. Tools such as VADER (Valence Aware Dictionary and Sentiment Reasoner) and LIWC (Linguistic Inquiry and Word Count) rely on predefined dictionaries of words associated with sentiment scores (Villanueva-Miranda et al., 2025). These models are interpretable and computationally efficient, making them suitable for large datasets. However, lexicon-based methods often fail to capture nuanced context, sarcasm, or implicit expressions of stress, limiting their predictive accuracy (Lee, 2025). For instance, a student's comment like "I can't even" may indicate high stress, yet a simple lexicon approach might classify it as neutral. Despite these limitations, lexicon-based methods continue to be widely used in mental health studies, particularly when interpretability is prioritized (Arslan, 2026).

**Figure 2.1:** Lexicon-based sentiment analysis

#### 2.1.3 Transformer-Based Models for Contextual Sentiment Detection

More recent studies have adopted transformer-based models such as BERT and RoBERTa to enhance sentiment analysis accuracy (Fahmi & Nababan, 2025). These deep learning architectures leverage contextual embeddings to understand relationships between words and phrases, enabling the detection of complex emotional patterns (Bello et al., 2023). Transformer models can recognize implicit expressions of stress, multi-sentiment sentences, and domain-specific language common in academic discourse (Mehreen et al., 2024). Empirical studies have shown that transformer-based classifiers outperform traditional lexicon methods in detecting nuanced psychological states from social media posts. While these models require greater computational resources and may be less interpretable, their ability to capture subtle emotional signals makes them highly suitable for forecasting student's well-being in online academic communities.

**Figure 2.2:** Transformer-based sentiment detection

#### 2.1.4 Applications of NLP in Student Mental Health Research

NLP techniques have been widely applied in academic and social media contexts to monitor mental health trends. Reddit subreddits such as r/college and r/gradschool serve as rich sources of student discourse, providing insights into stress patterns over academic semesters. Research indicates that sentiment fluctuations often align with academic milestones, including midterms and finals (Hu, 2012). Similar analyses of Twitter and Instagram reveal spikes in negative sentiment preceding high-stress periods. However, most of these studies focus on cross-sectional analysis, capturing only static snapshots of sentiment rather than dynamic temporal trends. This limitation underscores the need to integrate NLP-based sentiment analysis with time-series approaches for predictive modeling.

#### 2.1.5 Temporal Limitations of Existing NLP-Based Approaches

Despite the advances in NLP, a gap remains in temporal analysis of psychological signals (Rodríguez-Ibánez et al., 2023). Most studies report sentiment scores at a single point or average them over long periods, overlooking day-to-day or week-to-week variations. Additionally, sentiment is often treated as a continuous score rather than being converted into discrete counts, which are more appropriate for statistical modeling and forecasting (Nip & Berthelier, 2024). Addressing this gap requires the aggregation of NLP-classified stress mentions into time-indexed count data and applying appropriate statistical and forecasting models. This integration enables the development of predictive frameworks capable of anticipating stress surges, which is essential for proactive mental health interventions in university settings.

#### 2.1.6 Hybrid and Multimodal NLP Approaches for Predictive Monitoring

Recent studies have explored hybrid approaches that combine lexicon-based and transformer-based methods to balance interpretability and predictive performance (Horvat et al., 2024). Other emerging trends include multimodal analysis, integrating textual sentiment with temporal activity patterns, engagement metrics, or survey responses to enhance predictive capabilities (Dursun & Eken, 2024). These hybrid approaches demonstrate that NLP can evolve from simple classification to being part of a comprehensive predictive framework (Miah et al., 2024). Such developments provide a foundation for integrating NLP with statistical models and time-series forecasting, enabling proactive monitoring of student stress and supporting timely interventions.

### 2.2 Statistical Models for Count Data

#### 2.2.1 Poisson Regression and Its Limitations

After extracting stress-related posts using NLP, researchers commonly aggregate these into count data representing the number of high-stress mentions per day or week. Count data are discrete and non-negative, making Poisson regression a natural starting point for modeling (Lee et al., 2011). Poisson regression assumes that the mean and variance of the counts are equal, which is suitable for well-behaved datasets but often unrealistic for social media signals, which tend to be sparse and irregular (Bektashi et al., 2022). Nevertheless, Poisson models have been successfully applied in health and behavioral studies, demonstrating their utility for modeling discrete event counts (Fernandez & Vatcheva, 2022).

| Regression Model | When to Use | Examples in Student Mental Health Research |
|------------------|-------------|-------------------------------------------|
| **Poisson Regression** | • Daily or weekly counts of stress-related posts follow a Poisson distribution. • Mean-variance relationships are approximately equal. | • Counting daily high stress mentions in Reddit student subreddits. • Analyzing frequency of anxiety-related posts around midterm and final periods. |
| **Negative Binomial Regression** | • Count data exhibits overdispersion (variance > mean). • Mean-variance relationships are not equal. | • Modeling spikes in stress-related posts during exam weeks. • Investigating variations in discussion about academic workload across multiple subreddits. |

#### 2.2.3 Discrete Time Series Models for Count Data

Beyond static Poisson regression, discrete time series models for count data have been proposed to capture temporal dependence in longitudinal event counts. In social media contexts, stress-related posts often exhibit persistence and clustering, where elevated counts at one time point increase the likelihood of elevated counts at subsequent time points. Poisson autoregressive models extend the standard Poisson framework by allowing the expected count at a given time to depend on past observations, thereby modeling temporal autocorrelation while preserving the discrete and non-negative nature of the data (Fokianos, 2012). Related formulations, such as INGARCH models, further account for overdispersion and bursty dynamics commonly observed in online behavioral signals (Ferland et al., 2006). These approaches are particularly relevant for analyzing time-varying stress indicators derived from social media, as they explicitly model the evolution of emotional expression over time.

#### 2.2.4 Negative Binomial Models and Overdispersion

In many cases, social media count data to exhibit overdispersion, where variance exceeds the mean (Lindén & Mäntyniemi, 2011). This is especially common in Reddit posts, where activity may be minimal on typical days but spike sharply around exams or other academic events. Negative Binomial regression extends Poisson models to accommodate overdispersion, providing more reliable parameter estimates and improved model fit (Lindén & Mäntyniemi, 2011). Applications of Negative Binomial models in social media research and mental health studies demonstrate robust performance with sparse and highly variable count data (Erdi̇Nç et al., 2017). By modeling the relationship between temporal factors and stress counts, these approaches provide a statistical foundation for predictive frameworks.

#### 2.2.5 Covariate Integration for Improved Predictive Modelling

Beyond handling overdispersion, count models can incorporate covariates such as day of the week, academic events, or platform engagement metrics. Including these factors can improve model accuracy by accounting for predictable fluctuations in activity (Atkins et al., 2012). For example, prior research has incorporated exam schedules as covariates, capturing systematic changes in student stress mentions. Similarly, weekend effects and other recurring patterns have been shown to reduce residual errors in modeling (Klakattawi et al., 2018). Integrating NLP-derived counts with such covariates allows researchers to build richer models that bridge sentiment analysis and predictive forecasting.

#### 2.2.6 Limitations of Count Models and Need for Forecasting

Despite their utility, statistical models of count data face several challenges. Sparse data from smaller subreddits can lead to unstable estimates, while sudden spikes from viral posts can distort predictions. Careful preprocessing, data aggregation, and model selection are therefore critical. These limitations highlight the need for complementary time-series forecasting approaches to better capture temporal dependencies and provide actionable predictions.

#### 2.2.7 Zero-Inflated and Hurdle Models for Sparse Data

In addition to modelling overdispersion, zero-inflated and hurdle models explicitly address the structural absence of events in count data. In the context of student mental health discourse, zero counts may arise not only from low stress levels but also from reduced platform engagement or reluctance to express distress publicly. Zero-inflated models assume that zeros are generated by two distinct processes: one governing whether an observation is structurally zero and another governing the count process itself. Hurdle models adopt a similar logic by separating the binary occurrence of events from their frequency (Feng, 2021). Studies in public health surveillance and online behavioral analysis suggest that these models offer improved interpretability and predictive accuracy when applied to sparse social media datasets, although their adoption in NLP-based mental health research remains limited (Kassahun et al., 2014).

#### 2.2.8 Challenges in Incorporating Temporal and Contextual Covariates

Despite these advantages, incorporating temporal and contextual covariates introduces additional modelling challenges. Covariate effects may vary over time, particularly in academic settings where stressors are episodic rather than constant. For example, the impact of examination periods on stress-related posting may differ across semesters or institutions, limiting the generalizability of fixed-effect models. Moreover, multicollinearity between temporal variables, such as day-of-week effects and academic calendar indicators, can complicate parameter estimation (Ariens et al., 2022). These issues highlight the importance of careful feature selection and validation strategies when integrating contextual information into count-based models, especially when the objective shifts from explanation toward forecasting.

### 2.3 Time-Series Forecasting of Behavioral Signals

#### 2.3.1 Importance of Time-Series Forecasting in Behavioral Signals

Once count-based stress signals are derived, time-series forecasting enables prediction of future stress surges, facilitating proactive interventions. Behavioral signals often exhibit recurring patterns, such as weekly cycles, as well as spikes associated with academic deadlines (Soyiri & Reidpath, 2012). Accurate forecasting requires methods that account for these trends and seasonal effects. Furthermore, predictive insights can support university administrators in planning mental health resources more effectively, allowing interventions to be deployed before stress peaks (Tomov et al., 2023). Understanding temporal dynamics also provides researchers with a way to quantify the impact of academic schedules, holidays, or other contextual factors on student well-being, which is essential for both theoretical and practical applications.

#### 2.3.2 Classical Forecasting Models: ARIMA and SARIMA

Classical forecasting models, such as ARIMA (AutoRegressive Integrated Moving Average) and SARIMA (Seasonal ARIMA), are widely employed for univariate time-series data. ARIMA captures trends and autocorrelations in stationary sequences, while SARIMA accommodates seasonality, making it suitable for modeling fluctuations aligned with academic calendars. Applications in social media research have demonstrated ARIMA's ability to forecast user activity and sentiment trends, enabling short-term predictions of behavioral signals. However, classical methods assume linear relationships and may struggle with sudden spikes in count data, necessitating preprocessing or transformation (What Are the Main Challenges and Limitations of ARIMA Models in Practice?, 2023). Additionally, these models often require manual parameter tuning and stationarity testing, which can be time-consuming and may limit scalability when applied to multiple subreddits or large datasets over extended periods.

#### 2.3.3 Modern Forecasting Approaches: Prophet and LSTM

Modern forecasting approaches, including Prophet and LSTM (Long Short-Term Memory networks), address some limitations of classical models. Prophet is designed to automatically model seasonality and incorporate holiday or event effects, making it particularly suited for academic-related stress forecasting (Chugh, 2025). LSTM networks can capture long-term dependencies and non-linear temporal trends, although they require larger datasets and higher computational resources (Chaudhary, 2025). Studies applying Prophet to social media sentiment have shown its ability to forecast peaks in activity and negative sentiment, highlighting its applicability for monitoring student well-being. In addition, modern approaches allow for greater flexibility in incorporating external regressors, such as exam schedules or engagement metrics, which can improve predictive accuracy. These models also offer opportunities to experiment with hybrid architectures, combining statistical methods and neural networks for enhanced robustness in complex, non-stationary datasets (Lim et al., 2021).

#### 2.3.4 Gaps in Integrating NLP-Derived Counts with Forecasting

Despite the availability of advanced NLP, count-based statistical models, and forecasting techniques, few studies have integrated these components (Sattar et al., 2025). Research tends to focus on either sentiment analysis or time-series forecasting independently, without combining NLP-derived stress counts with predictive models. Integrating these methodologies is necessary to anticipate stress surges rather than merely describe historical trends (Weng et al., 2025). Developing such a framework offers the potential for proactive, actionable insights, providing university administrators with data-driven tools for mental health intervention planning. Moreover, integration can help uncover patterns that are not visible when sentiment and temporal data are analyzed separately, such as how sudden spikes in negative sentiment interact with recurring academic events. This gap indicates a critical opportunity for novel research that connects the strengths of NLP and forecasting for continuous monitoring of student well-being.

#### 2.3.5 Limitations and Challenges in Behavioral Signal Forecasting

Even with robust forecasting models, predicting student stress from social media data presents unique challenges. Behavioral signals are often noisy and influenced by external factors such as viral posts, sudden news, or changes in platform engagement, which classical and even modern models may fail to capture. Additionally, irregular posting behavior can create gaps in data, complicating trend detection. These factors necessitate careful preprocessing, smoothing techniques, or hybrid modeling strategies to ensure reliable predictions (Dealing With Erratic Data in Time Series Forecasting: Strategies and Algorithms, 2023). Furthermore, the reliance on historical data limits the ability of models to account for unprecedented events or changes in student behavior, emphasizing the importance of continuous model retraining and validation.

#### 2.3.6 Future Directions for Integrated Predictive Frameworks

A promising direction for future research is the development of integrated frameworks that combine NLP-derived sentiment counts with both statistical and machine learning forecasting models (Liu et al., 2025). Hybrid approaches, such as combining ARIMA with LSTM or embedding contextual academic events into Prophet models, could improve prediction of episodic stress surges (Mahmud et al., 2025). Moreover, such frameworks could allow real-time monitoring and early-warning systems for university mental health support, bridging the current gap between descriptive analysis and actionable forecasting. Additionally, integrating visualization tools and automated alerts could make these frameworks more practical for university administrators who may not have technical expertise. Finally, future studies should investigate model interpretability and ethical considerations, ensuring that predictive tools are both transparent and responsibly applied to support student well-being (Wu et al., 2025).

### 2.4 Challenges, Limitations, and Research Gaps

#### 2.4.1 Data Sparsity, Noise, and Irregularity

Although social media provides a rich data source, several challenges arise in using it for mental health research. Reddit posts may be sparse, irregular, or noisy, particularly in smaller subreddits where user activity is limited. Viral posts or sudden discussion spikes can create extreme outliers that distort statistical modeling and forecast accuracy. These irregularities complicate temporal trend analysis and make it difficult to distinguish genuine patterns from random fluctuations. Additionally, missing data or uneven posting behavior across days or weeks can reduce the robustness of predictive models, especially for smaller or less active student communities. Therefore, careful preprocessing, aggregation, and normalization of data are essential to ensure that analyses reflect genuine behavioral signals rather than artifacts of online activity.

#### 2.4.2 Methodological and Modeling Limitations

Count-based statistical models and time-series forecasting techniques, while powerful, also have inherent limitations. Poisson and Negative Binomial models may struggle with extreme overdispersion or structural zeros, while ARIMA, SARIMA, and LSTM models can be sensitive to non-stationary trends, missing data, or sudden spikes. Hybrid approaches can improve performance, but integrating multiple modeling methods increases computational complexity and requires careful parameter tuning and validation. Furthermore, the interpretability of advanced models, particularly deep learning approaches like LSTM, remains a significant challenge. Decision-makers, such as university administrators, need to understand why a model predicts a surge in student stress to implement timely interventions effectively (Diaa et al., 2024). These methodological limitations highlight the need for transparent, interpretable, and robust frameworks that balance predictive power with usability.

#### 2.4.3 Research Gaps and Future Directions

Despite advances in NLP, count-based modeling, and time-series forecasting, gaps remain in integrating these approaches into a coherent framework for student mental health prediction (Sattar et al., 2025). Most research analyzes sentiment or behavioral trends in isolation, overlooking the potential for predictive frameworks that combine NLP-derived stress counts with temporal forecasting models. Additionally, few studies address the complex interactions between academic calendars, user engagement patterns, and episodic stressors. There is also limited exploration of hybrid models that balance interpretability, accuracy, and computational efficiency. Addressing these gaps presents a significant opportunity to advance both theory and practice by creating proactive, evidence-based monitoring systems. Such systems could provide early-warning insights, guide resource allocation, and inform interventions to support student well-being, aligning with ethical research standards and institutional priorities.

#### 2.4.4 Challenges in Modeling Temporal Dynamics

A key challenge in existing research lies in accurately capturing temporal dynamics of stress-related sentiment counts. Student mental health signals are often influenced by complex, non-linear interactions between academic schedules, social events, and individual behavior. Standard forecasting models may fail to detect sudden, irregular surges, such as unexpected assignment deadlines or viral discussions, leading to prediction errors. Additionally, temporal dependencies across multiple subreddits or platforms are rarely considered, despite the fact that students may engage with multiple online communities simultaneously. Addressing these challenges requires novel approaches that combine temporal modeling with contextual covariates, ensuring predictions reflect the multifaceted nature of student stress patterns.

#### 2.4.5 Opportunities for Hybrid and Predictive Frameworks

The underexplored integration of NLP, statistical modeling, and time-series forecasting presents significant opportunities for future research. Hybrid frameworks that combine lexicon-based and transformer-based sentiment analysis with count models and advanced forecasting techniques could enhance both accuracy and interpretability (Sharma & Sharma, 2025). Such approaches would allow researchers to predict stress surges with higher confidence while maintaining transparency for stakeholders. Furthermore, embedding academic calendars, platform engagement metrics, and behavioral covariates into these models can improve the granularity and actionable value of predictions. Future studies adopting these hybrid approaches could establish a new standard for proactive mental health monitoring, bridging the current gap between descriptive analysis and real-time, evidence-based interventions.

---

## 3. Conclusion

The literature reviewed in this study underscores the growing importance of integrating computational methods with psychological research to monitor student mental health through social media (Gandy et al., 2024). Advances in Natural Language Processing (NLP) have enabled the extraction of meaningful signals from unstructured textual data, providing near real-time insights into emotional and behavioral states. Lexicon-based approaches such as VADER and LIWC have demonstrated utility in efficiently quantifying sentiment, while transformer-based models, including BERT and RoBERTa, offer enhanced capability to detect nuanced and context-dependent expressions of stress and anxiety. Hybrid approaches that combine lexicon-based interpretability with transformer-based predictive performance further extend the analytical toolkit, enabling researchers to capture both explicit and implicit indicators of psychological distress. These developments highlight the potential of NLP to move beyond traditional survey-based methods, offering continuous, scalable, and granular monitoring of student well-being.

Despite these advances, the translation of textual sentiment into actionable, predictive insights remains limited. Statistical models for count data, such as Poisson and Negative Binomial regression, provide a principled framework to model the frequency of stress-related posts, accounting for overdispersion and other distributional challenges. Zero-inflated and hurdle models address the structural absence of events in sparse datasets, enhancing interpretability and model fit. Furthermore, integrating temporal and contextual covariates—including day-of-week effects, academic calendars, and engagement metrics—can improve predictive performance and better capture the episodic nature of student stress. However, these models face challenges related to parameter estimation, multicollinearity, and robustness, particularly in smaller or irregular datasets. Collectively, these findings indicate that while count-based modeling is foundational, it is insufficient on its own to forecast stress surges accurately, necessitating complementary approaches.

Time-series forecasting methods offer a critical avenue to anticipate fluctuations in student mental health signals. Classical models such as ARIMA and SARIMA capture linear trends, autocorrelation, and seasonality, making them suitable for structured temporal data aligned with academic schedules. However, these approaches may struggle with non-linear patterns, sudden spikes, or long-term dependencies inherent in social media discourse. Modern methods, including Prophet and LSTM networks, address many of these limitations by automatically modeling seasonality, integrating event effects, and capturing non-linear temporal dependencies. Studies applying these approaches to social media sentiment demonstrate the feasibility of forecasting stress surges, yet research integrating NLP-derived counts with advanced forecasting remains sparse. This gap underscores the need for hybrid predictive frameworks that combine textual analysis, statistical modeling, and temporal forecasting, facilitating proactive and evidence-based mental health interventions.

Ethical and representativeness considerations remain central to the responsible use of social media data for mental health research. While Reddit and other platforms offer publicly available content, privacy and consent remain significant concerns. Anonymization and adherence to platform-specific terms of service are essential to protect user identities. Additionally, social media users may not represent the broader student population, leading to potential sampling bias. Certain groups may be underrepresented or overrepresented based on engagement patterns, limiting the generalizability of findings. Addressing these ethical and methodological challenges is critical to ensuring that predictive frameworks are both accurate and socially responsible.

Synthesizing insights from NLP, statistical modeling, and forecasting highlights several key gaps in the current literature. First, most studies remain descriptive, focusing on historical sentiment trends rather than predictive frameworks capable of anticipating future stress surges. Second, integration across methodological domains is limited, with few studies combining NLP-derived counts with robust statistical models and time-series forecasting. Third, challenges in sparse and irregular data, sudden spikes, and temporal dependencies are often insufficiently addressed. Fourth, ethical and generalizability concerns are frequently acknowledged but rarely operationalized in modeling frameworks. Addressing these gaps is essential for advancing the scientific understanding of student mental health and for developing tools that can support proactive interventions in real-world academic settings.

This literature review establishes a clear rationale for the proposed capstone project, which seeks to develop a predictive framework for monitoring student mental health through online discourse. By aggregating NLP-classified stress mentions into count-based signals and applying statistical models integrated with time-series forecasting, the project aims to anticipate periods of elevated psychological demand. Such a framework offers actionable insights for university administrators, enabling targeted resource allocation, timely interventions, and evidence-based program planning. Furthermore, this approach contributes methodologically by bridging traditionally separate research domains—text-based sentiment analysis, count-based statistical modeling, and predictive time-series analysis—thereby advancing the field of computational mental health monitoring.

In conclusion, the reviewed literature demonstrates both the potential and the limitations of existing methodologies for monitoring student mental health via social media. NLP provides the foundational tools for extracting psychological signals, statistical models contextualize these signals into actionable insights, and time-series forecasting offers the capacity to anticipate stress surges. Yet, significant gaps persist, particularly in integrating these methods, addressing ethical considerations, and capturing the complex temporal and contextual dynamics of student stress. The proposed research addresses these challenges, offering a novel, proactive, and data-driven framework that aligns with broader goals of mental health promotion and student well-being. By synthesizing theoretical and methodological advances, this capstone project not only fills critical gaps in the literature but also provides practical guidance for higher education institutions seeking to support students in a timely and evidence-based manner.

**Word Count: 4500**

---

## 4. References

(3) Dealing with Erratic Data in Time Series Forecasting: Strategies and Algorithms | LinkedIn. (2023, August 17). https://www.linkedin.com/pulse/dealing-erratic-data-time-series-forecasting-ayush-chauhan/

Ariens, S., Adolf, J. K., & Ceulemans, E. (2022). Collinearity Issues in Autoregressive Models with Time-Varying Serially Dependent Covariates. Multivariate Behavioral Research, 58(4), 687–705. https://doi.org/10.1080/00273171.2022.2095247

Arslan, E., PhD. (2026, January 20). Sentiment analysis methods in 2026. AIMultiple. https://research.aimultiple.com/sentiment-analysis-methods/

Atkins, D. C., Baldwin, S. A., Zheng, C., Gallop, R. J., & Neighbors, C. (2012). A tutorial on count regression and zero-altered count models for longitudinal substance use data. Psychology of Addictive Behaviors, 27(1), 166–177. https://doi.org/10.1037/a0029508

Bektashi, X., Rexhepi, S., & Limani–Bektashi, N. (2022). Dispersion of count Data: A case study of poisson distribution and its Limitations. Asian Journal of Probability and Statistics, 18–28. https://doi.org/10.9734/ajpas/2022/v19i230464

Bello, A., Ng, S., & Leung, M. (2023). A BERT framework to sentiment analysis of tweets. Sensors, 23(1), 506. https://doi.org/10.3390/s23010506

Chhabra, J., Pilkington, V., Benakovic, R., Wilson, M. J., La Sala, L., & Seidler, Z. (2025). Social Media and Youth Mental Health: Scoping Review of platform and policy recommendations. Journal of Medical Internet Research, 27, e72061. https://doi.org/10.2196/72061

Chugh, V. (2025, November 5). Facebook Prophet: A Modern approach to Time series Forecasting. https://www.datacamp.com/tutorial/facebook-prophet

Chaudhary, A. (2025). International Journal of Sciences and Innovation Engineering. International Journal of Sciences and Innovation Engineering. https://doi.org/10.70849/ijsci

Diaa, N. M., Ahmed, S. S., Salman, H. M., & Sajid, W. A. (2024). Statistical challenges in social media data analysis sentiment tracking and beyond. Journal of Ecohumanism, 3(5), 365–384. https://doi.org/10.62754/joe.v3i5.3912

Dursun, S., & Eken, S. (2024, November 14). Multimodal sentiment analysis in natural disaster data on social media. EUDL. https://eudl.eu/doi/10.4108/eetsc.5860

Feng, C. X. (2021). A comparison of zero-inflated and hurdle models for modeling zero-inflated count data. Journal of Statistical Distributions and Applications, 8(1), 8. https://doi.org/10.1186/s40488-021-00121-4

Fernandez, G. A., & Vatcheva, K. P. (2022). A comparison of statistical methods for modeling count data with an application to hospital length of stay. BMC Medical Research Methodology, 22(1), 211. https://doi.org/10.1186/s12874-022-01685-8

Gandy, L. M., Ivanitskaya, L. V., Bacon, L. L., & Bizri-Baryak, R. (2024). Public health discussions on social media: Evaluating Automated sentiment Analysis methods. JMIR Formative Research, 9, e57395. https://doi.org/10.2196/57395

GeeksforGeeks. (2025, July 23). ARIMA vs SARIMA Model. GeeksforGeeks. https://www.geeksforgeeks.org/machine-learning/arima-vs-sarima-model/

Horvat, M., Gledec, G., & Leontić, F. (2024). Hybrid Natural Language Processing Model for Sentiment Analysis during Natural Crisis. Electronics, 13(10), 1991. https://doi.org/10.3390/electronics13101991

Hu, W. (2012). Real-Time Twitter Sentiment toward Midterm Exams. Sociology Mind, 02(02), 177–184. https://doi.org/10.4236/sm.2012.22023

Kassahun, W., Neyens, T., Molenberghs, G., Faes, C., & Verbeke, G. (2014). Marginalized multilevel hurdle and zero‐inflated models for overdispersed and correlated count data with excess zeros. Statistics in Medicine, 33(25), 4402–4419. https://doi.org/10.1002/sim.6237

Klakattawi, H., Vinciotti, V., & Yu, K. (2018). A simple and adaptive dispersion regression model for count data. Entropy, 20(2), 142. https://doi.org/10.3390/e20020142

Ks, R. (2024). Analyzing Online Conversations on Reddit: A study of stress and anxiety through topic modeling and sentiment analysis. Cureus, 16(9), e69030. https://doi.org/10.7759/cureus.69030

Lee, J., Han, G., Fulp, W. J., & Giuliano, A. R. (2011). Analysis of overdispersed count data: application to the Human Papillomavirus Infection in Men (HIM) Study. Epidemiology and Infection, 140(6), 1087–1094. https://doi.org/10.1017/s095026881100166x

Lee, S. (2025). Sentiment Lexicons: A Comprehensive guide. https://www.numberanalytics.com/blog/sentiment-lexicons-computational-linguistics-guide

Lim, L. K. Y., Kong, Y. H., & Chin, W. Y. (2021). Forecasting Facebook User Engagement using Hybrid Prophet LSTM and iForest. https://iemjournal.com.my/index.php/iem/article/view/123

Lindén, A., & Mäntyniemi, S. (2011). Using the negative binomial distribution to model overdispersion in ecological count data. Ecology, 92(7), 1414–1421. https://doi.org/10.1890/10-1831.1

Liu, Z., Zhang, Z., & Zhang, W. (2025). A hybrid framework integrating traditional models and deep learning for Multi-Scale Time series forecasting. Entropy, 27(7), 695. https://doi.org/10.3390/e27070695

MacAvaney, S., Desmet, B., Cohan, A., Soldaini, L., Yates, A., Zirikly, A., & Goharian, N. (2018, June 20). RSDD-Time: Temporal Annotation of Self-Reported Mental Health Diagnoses. arXiv.org. https://arxiv.org/abs/1806.07916

Madrid‐Cagigal, A., Kealy, C., Potts, C., Mulvenna, M. D., Byrne, M., Barry, M. M., & Donohoe, G. (2025). Digital Mental Health Interventions for University Students With Mental Health Difficulties: A Systematic Review and Meta‐Analysis. Early Intervention in Psychiatry, 19(3), e70017. https://doi.org/10.1111/eip.70017

Mahmud, A., Noor, S. H. N. S. H., Musa, K. I., Hamzah, F. M., Yudin, Z. M., Kamaruddin, N., Madawana, A. M., & Nawi, M. a. A. (2025). Hybrid ARIMA-LSTM for COVID-19 forecasting: a comparative AI modeling study. PeerJ Computer Science, 11, e3195. https://doi.org/10.7717/peerj-cs.3195

Mehreen, F., Banbhrani, S. K., Akhter, M. N., & Noureen, F. (2024, November 30). SENTIMENT ANALYSIS OF SOCIAL MEDIA TEXT USING TRANSFORMER-BASED LANGUAGE MODELS: A STUDY ON PUBLIC OPINION MINING AND ITS APPLICATIONS IN DECISION-MAKING. https://policyrj.com/index.php/1/article/view/636

Miah, M. S. U., Kabir, M. M., Sarwar, T. B., Safran, M., Alfarhood, S., & Mridha, M. F. (2024). A multimodal approach to cross-lingual sentiment analysis with ensemble of transformer and LLM. Scientific Reports, 14(1), 9603. https://doi.org/10.1038/s41598-024-60210-7

Nip, J. Y. M., & Berthelier, B. (2024). Social Media sentiment analysis. Encyclopedia, 4(4), 1590–1598. https://doi.org/10.3390/encyclopedia4040104

Oryngozha, N., Shamoi, P., & Igali, A. (2024). Detection and analysis of Stress-Related posts in Reddit's Acamedic communities. IEEE Access, 12, 14932–14948. https://doi.org/10.1109/access.2024.3357662

Osman, W. A. (2025). Social media use and associated mental health indicators among University students: a cross-sectional study. Scientific Reports, 15(1), 9534. https://doi.org/10.1038/s41598-025-94355-w

Phillips, L., Dowling, C., Shaffer, K., Hodas, N., & Volkova, S. (2017, June 19). Using social media to Predict the Future: A Systematic Literature review. arXiv.org. https://arxiv.org/abs/1706.06134

Rodríguez-Ibánez, M., Casánez-Ventura, A., Castejón-Mateos, F., & Cuenca-Jiménez, P. (2023). A review on sentiment analysis from social media platforms. Expert Systems With Applications, 223, 119862. https://doi.org/10.1016/j.eswa.2023.119862

Saeed, Q. B., & Cha, Y. (2025). Multi-modal deep-attention-BiLSTM based early detection of mental health issues using social media posts. Scientific Reports, 15(1), 35152. https://doi.org/10.1038/s41598-025-19141-0

Saha, K., Yousuf, A., Boyd, R. L., Pennebaker, J. W., & De Choudhury, M. (2022). Social media discussions predict mental health consultations on college campuses. Scientific Reports, 12(1), 123. https://doi.org/10.1038/s41598-021-03423-4

Sattar, M. U., Hasan, R., Palaniappan, S., Mahmood, S., & Khan, H. W. (2025). Beyond Polarity: Forecasting Consumer Sentiment with Aspect- and Topic-Conditioned Time Series Models. Information, 16(8), 670. https://doi.org/10.3390/info16080670

Sharma, K., & Sharma, A. (2025). Hybrid Model for Predicting Mental Health from Social Media Insights. Zenodo (CERN European Organization for Nuclear Research). https://doi.org/10.5281/zenodo.18104910

Tomov, L., Chervenkov, L., Miteva, D. G., Batselova, H., & Velikova, T. (2023). Applications of time series analysis in epidemiology: Literature review and our experience during COVID-19 pandemic. World Journal of Clinical Cases, 11(29), 6974–6983. https://doi.org/10.12998/wjcc.v11.i29.6974

Villanueva-Miranda, I., Xie, Y., & Xiao, G. (2025). Sentiment analysis in public health: a systematic review of the current state, challenges, and future directions. Frontiers in Public Health, 13, 1609749. https://doi.org/10.3389/fpubh.2025.1609749

Weng, Y., Isleem, H. F., Hindi, K. E., & Ezugwu, A. E. (2025). Natural language processing for extracting consumer sentiment dynamics through multimodal social media analysis to predict microeconomic consumption pattern shifts. Journal of Big Data, 12(1). https://doi.org/10.1186/s40537-025-01315-2

What are the main challenges and limitations of ARIMA models in practice? (2023, September 1). https://www.linkedin.com/advice/1/what-main-challenges-limitations-arima-models-practice

Wu, W., Zhang, G., Tan, Z., Wang, Y., & Qi, H. (2025, May 2). Dual-ForeCaster: a multimodal time series model integrating descriptive and predictive texts. arXiv.org. https://arxiv.org/abs/2505.01135

Yang, X., & Li, G. (2025). Psychological and Behavioral Insights from Social Media users: Natural Language Processing–Based Quantitative Study on Mental Well-Being. JMIR Formative Research, 9, e60286. https://doi.org/10.2196/60286