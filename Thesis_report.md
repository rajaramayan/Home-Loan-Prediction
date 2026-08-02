# A Comparative Study of Machine Learning Algorithms for Home Loan Amount Prediction

---

## Abstract

The rapid expansion of the housing finance sector in developing economies such as Nepal has created a critical demand for automated, objective, and accurate decision-support tools for home loan appraisal. Traditional manual underwriting procedures are often time-consuming, prone to human subjectivity, and inconsistent across loan officers. This thesis presents an end-to-end comparative study of four supervised machine learning regression algorithms — **Linear Regression**, **Decision Tree Regressor**, **Random Forest Regressor**, and **K-Nearest Neighbors (KNN) Regressor** — for predicting approved home loan amounts.

The study utilizes a primary dataset of 250 approved home loan records from the Nepalese financial context, encompassing applicant demographics, household income, property valuations, interest rates, tenures, and Equated Monthly Installments (EMI). Prior to model training, domain-informed feature engineering was executed to directly embed Nepal Rastra Bank (NRB) regulatory directives into the feature space. Key engineered variables include **Total Monthly Income** (`TotalIncome`), **Maximum Affordable EMI** (`MaxEmi` = 50% DTI cap), **Maximum Loan Amount** (`MaxLoanAmount`, derived via time-value-of-money annuity compound formulas), and **Eligible Loan Amount** (`EligibleLoanAmount` = 70% LTV property cap). Data cleaning involved Interquartile Range (IQR) outlier removal ($n=224$ clean records) and $\text{log1p}$ natural logarithm transformations to eliminate positive skewness across monetary features.

All models were evaluated on an 80/20 train-test split ($n_{\text{test}} = 45$) and validated using 5-fold cross-validation ($k=5$). Evaluation metrics included $R^2$ (coefficient of determination), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE). Empirical results demonstrate that the **Random Forest Regressor** significantly outperformed all baseline algorithms, achieving a holdout **Test $R^2$ of 0.9367**, an exceptional **5-Fold Cross-Validation Mean $R^2$ of 0.9761**, the lowest **Test MAE of 145,848 NPR**, and the lowest **Test RMSE of 201,173 NPR**. Linear Regression placed second ($\text{Test } R^2 = 0.9339$), benefiting from pre-linearized regulatory features, followed by KNN ($R^2 = 0.9251$) and Decision Tree ($R^2 = 0.8874$). Feature importance analysis confirmed that regulatory bounds (`EligibleLoanAmount` at 41.2% and `MaxLoanAmount` at 32.8%) govern loan quantum determination.

The optimal Random Forest model (`random_regresser.joblib`) was deployed as an interactive, multi-page web application using **Streamlit**, providing instantaneous loan estimation, exploratory analytics, and model governance tools for risk managers and loan officers. This research demonstrates that combining domain-specific regulatory feature engineering with ensemble learning establishes an accurate, auditable, and production-ready framework for automated credit evaluation in housing finance.

---

**Keywords:** Home Loan Prediction, Machine Learning, Random Forest Regressor, Feature Engineering, Loan-to-Value (LTV) Ratio, Debt-to-Income (DTI) Cap, Amortization, Streamlit, Nepal Rastra Bank

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Literature Review](#2-literature-review)
3. [Dataset Description](#3-dataset-description)
4. [Methodology](#4-methodology)
5. [Experimental Results](#5-experimental-results)
6. [Discussion](#6-discussion)
7. [Conclusion and Future Work](#7-conclusion-and-future-work)
8. [References](#8-references)

---

## 1. Introduction

### 1.1 Background

Housing is one of the most fundamental human needs, and home ownership remains a primary
financial goal for a majority of households in developing economies including Nepal.
Financial institutions — commercial banks, development banks, and cooperative societies —
provide home loans as the primary instrument for enabling property purchases. The process
of determining how much loan an applicant qualifies for involves assessing multiple
interrelated factors: the applicant's income, the market value of the property, prevailing
interest rates, repayment capacity, and regulatory guidelines such as the Loan-to-Value
(LTV) ratio mandated by the Nepal Rastra Bank.

Traditionally, loan officers evaluate these factors manually, relying on institutional
guidelines and personal judgment. This approach is inherently subjective, slow, and
difficult to scale. Moreover, inconsistency in decision-making across officers can lead to
either over-lending — increasing the institution's credit risk — or under-lending, which
excludes creditworthy applicants from accessing funds they genuinely need.

### 1.2 Motivation

Machine learning offers a data-driven alternative to manual appraisal. By learning
patterns from historical approved loan records, a trained model can estimate the
appropriate loan amount for new applicants in a consistent, objective, and near-instant
manner. This not only reduces processing time but also provides a quantifiable basis for
lending decisions that can be audited and improved over time.

While classification-based approaches to loan approval prediction (approve / reject) have
been widely studied, regression-based loan amount prediction — estimating the exact
quantum of a loan — has received comparatively less attention, particularly in the context
of Nepalese housing finance data.

### 1.3 Objectives

The objectives of this study are:

1. To build and compare four machine learning regression models — Linear Regression,
   Decision Tree Regressor, Random Forest Regressor, and K-Nearest Neighbors Regressor —
   for predicting the home loan amount.
2. To engineer domain-relevant financial features from raw applicant and property data
   that improve model predictive power.
3. To evaluate models rigorously using both hold-out test metrics and k-fold
   cross-validation to ensure generalisability.
4. To identify the best-performing model and deploy it as an interactive web application
   for real-time loan amount prediction.

### 1.4 Scope and Limitations

The study is scoped to home loans only. The dataset consists exclusively of approved
applications, meaning the models learn the distribution of approved loan amounts rather
than modelling the approval decision itself. The dataset contains 250 records, which is
sufficient for a comparative study but limits the statistical power of the conclusions.
The results are specific to the Nepalese financial context and may not generalise directly
to other countries with different regulatory frameworks or income distributions.

---

**Chapter 2: Literature Review**

**2.1 Introduction to Home Loan Prediction**

The global financial landscape has undergone a paradigm shift in the
assessment of creditworthiness and loan amount determination. Home loan
prediction, a critical function of retail banking, involves evaluating a
borrower\'s ability to repay a long-term debt secured by real estate.
Historically, this process was governed by the \"Five Cs of Credit\":
Character, Capacity, Capital, Collateral, and Conditions. However, the
manual interpretation of these factors often led to subjective biases,
operational inefficiencies, and a lack of scalability \[1\], \[13\].

In the contemporary era of Big Data, the volume and variety of financial
data points have grown exponentially, encompassing not just traditional
credit scores and income levels, but also alternative data such as
utility payments, transactional behavior, and even psychometric
insights. The core objective of home loan prediction is two-fold:
identifying eligible candidates (classification) and determining the
optimal loan quantum that balances the bank\'s risk appetite with the
borrower\'s needs (regression) \[15\], \[22\]. The precision of these
predictions is paramount; over-estimation leads to increased default
risk, while under-estimation results in missed business opportunities
and customer dissatisfaction.

2.2 Evolution of Machine Learning in Banking

The evolution of ML in banking represents a shift from descriptive
analytics to predictive and prescriptive frameworks. Early systems
primarily utilized logistic regression for credit scoring due to its
interpretability \[4\], \[21\]. However, as data volumes grew and
alternative data sources became available, more sophisticated algorithms
were adopted. The transition moved through decision trees and random
forests, which offered better handling of non-linear relationships,
towards advanced ensemble methods like XGBoost, LightGBM, and CatBoost
\[2\], \[10\]. Recent advancements (2020-2025) have seen the integration
of deep learning architectures and hybrid models that combine the
strengths of diverse algorithmic families to achieve unprecedented
accuracy in loan forecasting \[1\], \[25\].

2.3 Machine Learning Applications in Financial Services

The integration of Machine Learning (ML) in financial services has
transcended basic credit scoring. Current applications include fraud
detection, customer segmentation, personalized marketing, and automated
wealth management \[5\], \[19\]. In the context of lending, these
technologies are used to analyze high-dimensional sparse data,
synthesize transactional and social features, and mitigate information
asymmetry between borrowers and lenders \[19\], \[20\]. Furthermore,
ML-driven systems promote financial inclusion by leveraging alternative
data to assess the creditworthiness of underbanked populations, thereby
expanding the reach of formal banking services \[5\], \[23\].

2.4 Loan Eligibility and Amount Prediction

The core challenge in home loan prediction lies in balancing the binary
decision of eligibility with the continuous estimation of the loan
amount. Research by Shah \[22\] demonstrates the effectiveness of
multi-task frameworks, where LightGBM was found to be superior for
classification tasks (96.23% accuracy), while CatBoost excelled in
regression for value prediction (R2 = 0.8820). Similarly, Saha et al.
\[12\] highlight the deployment of real-time cloud-based systems using
XGBoost to handle numerical features like debt-to-income ratio and FICO
scores for immediate loan assessment. The integration of regression
techniques allows banks to tailor loan offers to individual borrower
capacities, optimizing the loan-to-value ratio \[18\], \[25\].

2.5 Credit Risk Assessment and Scoring

Credit risk assessment remains the fundamental driver for determining
loan amounts. Recent studies emphasize the importance of robust feature
engineering to identify key risk indicators. Nalini and Viswadhanush
\[15\] identified TotalDebtToIncomeRatio, BankruptcyHistory, and
CreditScore as the most dominant variables in predicting risk scores.
Advanced models like the one proposed by Wu \[4\] utilize cost-sensitive
learning and threshold tuning to reduce expected losses, achieving an
AUC of 0.95. The shift towards AI-driven underwriting has led to
substantial improvements in decision accuracy and fairness across
diverse demographic groups \[20\].

2.6 Regression and Classification Algorithms in Banking

Banking institutions utilize a spectrum of algorithms depending on the
task requirements for transparency versus performance.

•   \*\*Logistic Regression:\*\* Frequently used as a baseline for its
interpretability and solid performance in binary classification tasks
like loan approval \[4\], \[21\].

•   \*\*Decision Trees (DT):\*\* Offer simple rules for decision-making
but are prone to overfitting \[6\], \[17\].

•   \*\*Random Forest (RF):\*\* An ensemble approach that reduces
variance and improves robustness. It has been cited as a top performer
for risk score prediction, achieving R-squared values around 0.87
\[15\].

•   \*\*Support Vector Machines (SVM):\*\* Used for their ability to
handle high-dimensional data, though they are often resource-intensive
compared to boosting methods \[21\].

•   \*\*Gradient Boosting (XGBoost/LightGBM):\*\* Preferred for
high-accuracy tasks, effectively capturing non-linear feature
interactions \[10\], \[22\].

2.7 Ensemble and Deep Learning Approaches

Ensemble methods have consistently outperformed single-model approaches
in recent literature. Nguyen \[10\] found that XGBoost and LightGBM
achieved 98% accuracy in distinguishing between \"good\" and \"bad\"
loans, significantly outperforming traditional logit models. Deep
learning models, including Long Short-Term Memory (LSTM) networks and
Temporal Convolutional Networks (TCN), have also been explored for their
ability to capture temporal dependencies in borrower behavior, with TCNs
showing superior detection of default events \[2\]. Hybrid constructs
that combine supervised learning with deep learning architectures are
increasingly favored for their ability to exploit non-linear
relationships in complex financial datasets \[1\].

2.8 Explainable Artificial Intelligence (XAI)

As ML models become more complex, the need for transparency becomes
critical due to regulatory requirements like the GDPR. Explainable AI
(XAI) modules, such as SHAP (SHapley Additive exPlanations) and LIME
(Local Interpretable Model-agnostic Explanations), are being integrated
into loan prediction systems \[17\], \[26\]. These tools allow banks to
provide clear reasons for loan rejections, ensuring fairness and
building trust with applicants \[1\], \[5\]. Research indicates that XAI
helps balance high performance with the interpretability needed for
responsible AI application in the financial sector \[24\].

2.9 Data Preprocessing and Feature Engineering

Effective preprocessing is vital for model reliability. Common steps
include:

•   \*\*Handling Missing Values:\*\* Techniques such as KNNImputer or
mode imputation are standard \[9\], \[10\].

•   \*\*Class Imbalance:\*\* Given that loan defaults are relatively
rare, methods like SMOTE (Synthetic Minority Over-sampling Technique)
are frequently employed to balance datasets and prevent model bias
\[9\], \[17\].

•   \*\*Feature Engineering:\*\* This involves selecting
high-discriminative variables using Information Value (IV) or Kernel
Density Estimation (KDE) \[18\]. Innovative features such as time
deposits, online banking usage, and macroeconomic indicators
(unemployment, inflation) are increasingly integrated to provide a
holistic view of the borrower \[6\], \[10\], \[19\].

2.10 Model Optimization and Evaluation Metrics

Optimization techniques like Bayesian optimization and grid search are
used to fine-tune hyperparameters (e.g., learning rate, max depth)
\[11\], \[22\]. Performance is evaluated using a variety of metrics:

•   \*\*Classification:\*\* Accuracy, Precision, Recall, F1-score, and
AUC (Area Under the Curve) \[9\], \[21\].

•   \*\*Regression:\*\* Mean Absolute Error (MAE), Root Mean Square
Error (RMSE), and the Coefficient of Determination (R2 score) \[11\],
\[22\].

Higher recall is often prioritized in bank marketing to avoid missing
potential customers, while precision is critical for risk assessment to
minimize defaults \[6\], \[17\].

2.11 Comparison of Performance and Datasets

Comparative studies show a clear trend: ensemble boosting models like
XGBoost and LightGBM generally lead in performance across various
datasets. For instance, in a study comparing six models, XGBoost
achieved an R2 of 0.932 for property price estimation \[11\]. Datasets
vary from Kaggle\'s \"Home Credit Default Risk\" (containing hundreds of
thousands of rows) to bank-specific historical data \[9\], \[18\]. The
choice of model often depends on the specific balance required between
predictive power and computational resource intensity \[21\].

2.12 Critical Analysis of Research Gaps (Detailed)

Based on a systematic review of literature between 2020 and 2025,
several critical research gaps have been identified that directly inform
the motivation for this study:

2.12.1 Regression vs. Classification Task Formulation

A primary gap in the current body of research is the disproportionate
focus on classification over regression. The vast majority of studies
(approx. 80-90%) frame the loan problem as a binary classification
task---predicting either loan approval/rejection or default/no-default
\[2\], \[4\], \[7\], \[10\]. Consequently, the task of predicting the
continuous \*\*loan amount\*\* (regression) remains significantly
under-researched. Furthermore, there is a notable absence of multi-task
learning frameworks that could simultaneously optimize for eligibility
and quantum, which would better reflect the real-world operational
workflow of bank loan departments \[22\], \[25\].

2.12.2 Absence of Nepalese and South Asian Housing Finance Studies

A significant geographic and contextual gap exists. Among the reviewed
papers from 2020 to 2025, there is a distinct absence of studies
focusing on the \*\*Nepalese housing finance market\*\* or broader South
Asian housing economies. Most research utilizes datasets from Western or
global contexts (e.g., Lending Club, Freddie Mac, or generic Kaggle
datasets) \[3\], \[5\], \[8\], \[9\]. This regional gap means that
current ML advancements are not yet tailored to the unique regulatory,
economic, and socio-demographic factors that influence housing finance
in Nepal, such as high interest rate volatility and informal income
streams.

2.12.3 Regulatory Constraint Embedding in Feature Engineering

Current machine learning models in banking are predominantly
data-driven, prioritizing predictive accuracy through standard feature
engineering techniques like scaling and encoding \[1\], \[6\], \[10\].
However, they often fail to \*\*explicitly embed regulatory
constraints\*\*---such as Loan-to-Value (LTV) ratio caps, Debt Service
Ratio (DSR) limits, or central bank liquidity mandates---directly into
the feature engineering process or model architecture. These constraints
are typically treated as post-hoc filters rather than structural
components of the predictive engine \[2\], \[8\].

2.12.4 Rigorous Evaluation of Simple Baselines in Regression Setting

In the context of loan amount prediction, there is a lack of
\*\*rigorous evaluation of simple baselines\*\*. While sophisticated
ensemble methods like XGBoost and LightGBM are frequently compared
against each other, they are rarely benchmarked against regularized
linear models (e.g., Ridge, Lasso, or ElasticNet) in a way that
quantifies the actual marginal utility of the extra complexity for the
\*amount\* estimation task. Without these rigorous benchmarks, the
necessity for high-complexity models in banking---where simplicity is
valued for auditing---remains poorly justified for regression tasks
\[6\], \[14\], \[25\].

2.12.5 End-to-End Deployment for Non-Technical Practitioners

Finally, there is a distinct lack of research focusing on the
\*\*usability and end-to-end deployment\*\* of these models for
non-technical practitioners, such as bank loan officers. While papers
discuss model transparency through XAI modules (SHAP/LIME) \[1\],
\[17\], \[26\], they rarely detail the integration of these models into
low-code/no-code interfaces or decision-support systems that would
enable a non-technical user to interact with the model directly during a
customer consultation \[12\], \[20\].

2.13 Current Research Trends (2020-2025)

Recent trends include the adoption of federated learning,
transformer-based models, and Auto ML with Neural Architecture Search to
streamline model development \[3\]. There is a growing focus on ethical
AI, ensuring fairness across demographic groups, and complying with
evolving financial regulations \[5\], \[20\]. The shift toward hybrid
architectures that merge traditional statistical methods with advanced
deep learning represents the current frontier in loan prediction
research \[1\], \[19\].

2.14 Motivation for the Present Study

The motivation for this thesis is derived directly from these identified
gaps. By focusing on \*\*Home Loan Amount Prediction\*\* (regression)
using regional housing data, and rigorously benchmarking
state-of-the-art ensembles against regularized linear baselines while
embedding regulatory constraints (LTV/DSR) into the feature set, this
study aims to provide a deployment-ready framework for the banking
sector. This study is systematically address these gaps in the context
of the South Asian financial environment.

2.15 Summary

The literature review confirms that machine learning has fundamentally
reshaped home loan prediction. Ensemble boosting models like XGBoost,
LightGBM, and CatBoost have established themselves as the benchmarks for
accuracy and efficiency. However, challenges regarding data imbalance,
model interpretability, and the integration of diverse features remain.
This thesis builds upon these findings to develop a specialized model
for home loan amount prediction and particularly how a few four popular
ML algorithms compare with regard to performance metrics.

Table 2.1: Comparison of Existing Studies

  -------------------------------------------------------------------------------
  **Author(s)**   **Year**   **Objective**    **Best          **Performance
                                              Performing      Metric**
                                              Model**         
  --------------- ---------- ---------------- --------------- -------------------
  :\-\--          :\-\--     :\-\--           :\-\--          :\-\--

  Suraksha et al. 2025       Transparent Loan Hybrid ML/DL +  High
                             Risk Assessment  XAI             Accuracy/Recall

  Dulgerov        2025       Predicting Loan  TCN (Deep       Optimal
                             Defaults         Learning)       Precision/Recall

  Wu              2025       Cost-Sensitive   XGBoost         AUC: 0.95
                             Default                          
                             Prediction                       

  Teh & Ng        2025       Loan Default     LightGBM        Accuracy: 0.9764
                             Influencing                      
                             Features                         

  Nguyen          2025       Advanced         XGBoost /       Accuracy: 98%
                             Boosting for     LightGBM        
                             Defaults                         

  Dritsas et al.  2025       Residential      XGBoost         R2: 0.932
                             Price Estimation                 

  Saha et al.     2025       Real-Time Loan   XGBoost         Highest Accuracy
                             Approval                         

  Shah            2025       AI Financial     LightGBM (Cls)  Acc: 96.23% / R2:
                             Inclusion Model  / CatBoost      0.8820
                                              (Reg)           

  Hossain et al.  2025       Banking Credit   XGBoost         AUC: 91.3%
                             Risk Comparison                  

  Wang            2025       Bank Marketing   XGBoost         AUC: 90%
                             Prediction                       
  -------------------------------------------------------------------------------

 

Table 2.2: Comparison of Machine Learning Algorithms

  ----------------------------------------------------------------------------
  **Algorithm**   **Advantages**      **Disadvantages**    **Suitable
                                                           Applications**
  --------------- ------------------- -------------------- -------------------
  :\-\--          :\-\--              :\-\--               :\-\--

  Logistic        Statistical, high   Fails with           Small datasets,
  Regression      interpretability,   non-linear data.     simple scoring.
                  baseline for binary                      
                  tasks.                                   

  Random Forest   Ensemble (Bagging), Can be slow to       Risk scoring,
                  robust to outliers, train, less accurate general
                  good for risk       than boosting.       classification.
                  scores.                                  

  XGBoost         Gradient Boosting,  Computationally      Large datasets,
                  parallel            intensive,           competition-level
                  processing, handles \"black-box.\"       prediction.
                  sparse data.                             

  LightGBM        Leaf-wise growth,   Sensitive to small   Massive datasets,
                  fast training, high data (overfitting).  real-time
                  accuracy on large                        inference.
                  data.                                    

  CatBoost        Handles categorical Slower inference     Data with many
                  features natively,  than LightGBM.       categorical
                  reduces                                  variables.
                  overfitting.                             

  ANN / Deep      Captures non-linear Very complex,        Behavioral scoring,
  Learning        and temporal        requires sequential  default timelines.
                  dependencies        data.                
                  (LSTM/TCN).                              
  ----------------------------------------------------------------------------

 

Table 2.3: Research Gap Analysis

  ------------------------------------------------------------------------
  **Existing Study Gap**   **Limitation**            **Proposed Solution**
  ------------------------ ------------------------- ---------------------
  :\-\--                   :\-\--                    :\-\--

  \*\*Task Formulation\*\* Predominance of binary    Comparative analysis
                           classification (Yes/No).  of high-precision
                                                     regression models.

  \*\*Regional Scope\*\*   Absence of South          Utilization of
                           Asian/Nepalese studies.   regional housing
                                                     finance data context.

  \*\*Regulatory           Purely data-driven; no    Embedding LTV and DSR
  Integration\*\*          structural constraints.   limits in feature
                                                     engineering.

  \*\*Baseline Rigor\*\*   Lack of regularized       Detailed comparison
                           linear benchmarks.        of OLS, Ridge, and
                                                     Lasso vs Ensembles.

  \*\*Practitioner         Focus on math over        Framework for
  Usability\*\*            deployment/UI.            officer-centric
                                                     decision support.
  ------------------------------------------------------------------------

 

Table 2.4: Summary of Recent Studies (Dataset and Metrics)

  -------------------------------------------------------------------------
  **Author(s)**   **Dataset Source / **Primary Features    **Evaluation
                  Size**             Used**                Metrics**
  --------------- ------------------ --------------------- ----------------
  :\-\--          :\-\--             :\-\--                :\-\--

  Han Wu          Public Credit Risk Standardization &     ROC AUC, PR AUC,
                  Dataset            Encoding              Brier

  Neo Bank        5,000 Customers    Bank Relationships,   Recall
                                     Online Usage          

  Billah et al.   Kaggle (1,460      OverallQual, TotalSF, RMSE, MAE, R2
                  Homes)             GrLivArea             

  Teh & Ng        Kaggle (148,670    Interest, Credit      Acc, Pre, Rec,
                  rows)              Type, Spread          F1, AUC

  C. Nguyen       Freddie Mac        FICO, CLTV,           Accuracy,
                  (\>100k loans)     Inflation, Unemp      Sensitivity, AUC

  Nalini et al.   Synthetic Kaggle   DebtToIncome,         Accuracy,
                  (20k rows)         Bankruptcy, Score     R-squared

  Kai Wang        Kaggle Home Credit Credit Scores,        Pre, Rec, F1,
                  Default            Income, Amount        AUC

  Sanjiv J. Shah  Loan Approval      Not Specified         F1, Rec, Acc,
                  Dataset            (AI-Powered)          Pre, MAE, R2

  Hossain et al.  Demographic &      History, Income,      Acc, Pre, Rec,
                  Financial Data     Demographics          F1, AUC

  Saha et al.     Structured         FICO, Income, DTI,    Accuracy
                  Financial Data     Balance               
  -------------------------------------------------------------------------

 


---

## 3. Dataset Description

### 3.1 Source and Size

The dataset comprises **250 records** of approved home loan applications in Nepal. Each
record represents a single loan application and includes demographic, financial, and
loan-specific attributes.

### 3.2 Columns and Units

| Column          | Type        | Unit / Values                          | Notes                                  |
|-----------------|-------------|----------------------------------------|----------------------------------------|
| Gender          | Categorical | Male / Female                          |                                        |
| MartialStatus   | Categorical | Married / Unmarried                    |                                        |
| Age             | Numerical   | Years (range: 23–58)                   |                                        |
| Education       | Categorical | SLC / Plus Two / Bachelor / Master     | Plus Two is the most common (144/250)  |
| ClientIncome    | Numerical   | रू per year (annual)                   | Mean ≈ रू6,71,402/yr (≈ रू55,950/mo)  |
| FamilyIncome    | Numerical   | रू per year (annual)                   | Zero for 194 of 250 rows               |
| LoanType        | Categorical | Home (all 250 rows)                    | Dataset is home-loan specific          |
| HomeValue       | Numerical   | रू (Nepalese Rupees)                   | Market value of the property           |
| Emi             | Numerical   | रू per month                           | Monthly installment for this loan      |
| InterestRate    | Numerical   | % per annum (range: ~9–15%)            |                                        |
| Tenure          | Numerical   | Months (120 or 180 only)               | 10-year or 15-year loan term           |
| LoanStatus      | Categorical | Approved (all 250 rows)                | All applications in dataset approved   |
| LoanAmount      | Numerical   | रू (Nepalese Rupees)                   | **Target variable** (range: ~9.66L–60L)|

### 3.3 Key Observations

- All loans in the dataset are approved and of type "Home", making this a pure regression
  dataset with no classification component.
- `ClientIncome` and `FamilyIncome` are annual figures; the model converts them to monthly
  by dividing by 12.
- The `Emi` column stores the monthly installment corresponding to the approved
  `LoanAmount`, verified against the standard amortization formula:

$$EMI = \frac{L \cdot r \cdot (1+r)^n}{(1+r)^n - 1}$$

  where $L$ = loan amount, $r$ = monthly interest rate ($\text{InterestRate} / 1200$),
  and $n$ = tenure in months. The formula reproduces all `Emi` values exactly to two
  decimal places.

- `FamilyIncome` has a highly skewed distribution (median = 0; only 56 of 250 applicants
  report non-zero family income), suggesting most applicants rely solely on their own
  income for loan qualification.

---

## 4. Methodology

### 4.1 Overview

The methodology follows a standard supervised machine learning pipeline:

```
Raw Data → Preprocessing → Feature Engineering → Model Training →
Evaluation → Model Selection → Deployment
```

### 4.2 Data Preprocessing

**Step 1 — Income Aggregation**
ClientIncome and FamilyIncome (both annual) are summed and divided by 12 to produce a
single `TotalIncome` feature representing the applicant's total household monthly income.
The two original columns are then dropped.

**Step 2 — Outlier Removal**
Outliers in `HomeValue` and `TotalIncome` are identified and removed using the
Interquartile Range (IQR) method with a 1.5× fence:

$$\text{Lower fence} = Q_1 - 1.5 \times IQR, \quad \text{Upper fence} = Q_3 + 1.5 \times IQR$$

This step is applied sequentially — first on `HomeValue`, then on `TotalIncome` — to
prevent extreme values from distorting model training.

**Step 3 — Tenure Encoding**
The `Tenure` column takes only two values (120 months or 180 months). It is binary-encoded
as 0 (120 months) and 1 (180 months).

**Step 4 — Log Transformation**
Five monetary features (`HomeValue`, `TotalIncome`, `MaxEmi`, `MaxLoanAmount`,
`EligibleLoanAmount`) exhibit right-skewed distributions. `log1p` transformation
(i.e., $\log(1 + x)$) is applied to compress the range and approximate normality, which
benefits linear models and distance-based models such as KNN.

### 4.3 Feature Engineering

Three financially motivated features are derived from the preprocessed data:

**MaxEmi** — Maximum Affordable EMI
$$MaxEmi = \frac{TotalIncome}{2}$$
Banks in Nepal typically allow borrowers to commit at most 50% of their monthly income
toward loan repayments. This ratio (Debt-to-Income, or DTI cap) is the basis for
determining repayment capacity.

**MaxLoanAmount** — Income-Based Maximum Loan
$$MaxLoanAmount = MaxEmi \times \frac{(1+r)^n - 1}{r \cdot (1+r)^n}$$
This is the inverse of the EMI formula: given the maximum EMI a borrower can afford,
the formula back-calculates the largest principal that can be repaid over the given tenure
at the given interest rate. Here $r = \text{InterestRate} / 1200$ and $n$ = tenure months.

**EligibleLoanAmount** — Property-Based Maximum Loan (LTV Cap)
$$EligibleLoanAmount = 0.70 \times HomeValue$$
Nepal Rastra Bank guidelines require banks to limit home loan disbursements to at most
70% of the property's assessed market value, protecting the institution against
collateral risk.

### 4.4 Final Feature Set

The seven features used to train all models are:

| Feature              | Description                                      |
|----------------------|--------------------------------------------------|
| HomeValue            | Market value of the property (log-transformed)   |
| InterestRate         | Annual interest rate (%)                         |
| Tenure               | Loan term (encoded: 0 = 120 mo, 1 = 180 mo)     |
| TotalIncome          | Total monthly household income (log-transformed) |
| MaxEmi               | Maximum affordable monthly EMI (log-transformed) |
| MaxLoanAmount        | Income-based maximum loan (log-transformed)      |
| EligibleLoanAmount   | LTV-based maximum loan (log-transformed)         |

### 4.5 Models

Four regression algorithms are trained and compared:

**1. Linear Regression**
A baseline parametric model that fits a hyperplane to the training data by minimising the
sum of squared residuals. Assumes a linear relationship between features and the target.

**2. Decision Tree Regressor**
A non-parametric tree-based model that recursively partitions the feature space into
axis-aligned rectangles and predicts the mean target value within each leaf node.
Prone to overfitting on small datasets.

**3. Random Forest Regressor**
An ensemble of decision trees trained on bootstrap samples of the data with random
feature subsets at each split (bagging + feature randomness). Predictions are the average
of all tree predictions. Reduces the variance of individual decision trees and is robust
to overfitting.

**4. K-Nearest Neighbors (KNN) Regressor**
A non-parametric instance-based method that predicts the target as the mean of the $k$
nearest training samples in feature space (Euclidean distance). Set to $k = 5$ in this
study.

### 4.6 Evaluation Metrics

All models are evaluated using three metrics:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| R² | $1 - \frac{SS_{res}}{SS_{tot}}$ | Proportion of variance explained; higher is better |
| RMSE | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | Error in रू; penalises large errors; lower is better |
| MAE | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | Average absolute error in रू; lower is better |

Both **hold-out test** metrics (80/20 split) and **5-fold cross-validation** metrics are
reported to distinguish between test-set performance and generalisation ability.

---

## 5. Experimental Results

### 5.1 Exploratory Data Analysis & Feature Distribution Visualizations

Before model training, exploratory data analysis was conducted on `home_loan_data (1).csv` to inspect applicant demographics, collateral distributions, and bivariate relationships. 

![Figure 5.1: Marital Status Distribution](images/01_marital_status_distribution.png)
*Figure 5.1: Distribution of applicant marital status in the dataset.*

As shown in **Figure 5.1**, married applicants constitute approximately 65% of all loan applications (164 applicants), while unmarried applicants account for 35% (88 applicants). This reflects common underwriting observations where joint family income and marital status serve as key stability indicators.

![Figure 5.2: Education Level Distribution](images/02_education_distribution.png)
*Figure 5.2: Distribution of applicant education levels.*

**Figure 5.2** highlights that the majority of applicants hold intermediate/diploma (*Plus two*, 114 applicants) or *Bachelor* degrees (62 applicants), followed by *SLC* and *Master's* degree holders.

![Figure 5.3: Age Group Distribution](images/03_age_group_distribution.png)
*Figure 5.3: Demographic segmentation of loan applicants by age group.*

The age group distribution (**Figure 5.3**) demonstrates that primary home loan demand is concentrated in the `21-30` age bracket (~42%) and `31-40` age bracket (~35%), representing early and mid-career working individuals seeking initial home ownership.

![Figure 5.4: Home Value Distribution Binning](images/04_home_value_distribution.png)
*Figure 5.4: Five-bin frequency distribution of collateral home values.*

**Figure 5.4** illustrates the distribution of property collateral values (`HomeValue`) divided into five equal-width intervals. The vast majority of properties range between 1.7M and 5.5M NPR, with high-end luxury residential properties extending up to 9.0M NPR forming a right-skewed tail.

---

### 5.2 Outlier Analysis and Log Transformation

To ensure dataset quality and prevent extreme values from distorting regression models, Interquartile Range (IQR) filtering was applied to `HomeValue` and `TotalIncome`.

![Figure 5.5: HomeValue Outlier Detection (Before and After IQR Cleaning)](images/16_boxplot_HomeValue_with_outliers.png)
![Figure 5.6: HomeValue After Removing Outliers](images/17_boxplot_HomeValue_after_outliers.png)
*Figures 5.5 & 5.6: Box plots of `HomeValue` before and after removing upper-whisker outliers ($Q3 + 1.5 \times IQR$).*

As depicted in **Figures 5.5 and 5.6**, initial property valuations contained upper-tail outliers exceeding 7.5M NPR. Filtering records above the upper IQR threshold produced a clean, unimodal distribution (`df1`). Similarly, **Figures 5.7 and 5.8** (in `images/18_boxplot_TotalIncome_with_outliers.png` and `images/19_boxplot_TotalIncome_after_outliers.png`) demonstrate the removal of extreme total income outliers, yielding the final cleaned dataset ($n = 224$).

![Figure 5.7: Total Income Distribution Histogram](images/21_total_income_histogram.png)
*Figure 5.7: Frequency histogram of raw combined monthly family income (`TotalIncome`).*

Prior to log transformation, `TotalIncome` exhibited substantial positive skewness ($\text{skew} \approx 2.45$, **Figure 5.7**). Applying the natural logarithm transformation $\text{log1p}(x) = \ln(1+x)$ reduced skewness to $< 0.45$, stabilizing variance across monetary features (`HomeValue`, `TotalIncome`, `MaxEmi`, `MaxLoanAmount`, `EligibleLoanAmount`).

---

### 5.3 Feature Correlation & Collinearity Analysis

![Figure 5.8: Correlation Matrix Heatmap](images/20_correlation_matrix.png)
*Figure 5.8: Annotated Pearson correlation matrix heatmap for numeric features.*

The correlation matrix (**Figure 5.8**) confirms strong linear dependencies between engineered financial constraints and the target variable `LoanAmount`:
- **`EligibleLoanAmount` vs `LoanAmount`**: $\mathbf{r = 0.94}$, confirming that LTV collateral caps act as the primary upper ceiling.
- **`MaxLoanAmount` vs `LoanAmount`**: $\mathbf{r = 0.91}$, confirming the binding nature of income-derived debt service limits.
- **`HomeValue` vs `LoanAmount`**: $\mathbf{r = 0.92}$, demonstrating high property valuation sensitivity.

---

### 5.4 Quantitative Model Evaluation & Algorithm Comparison

All four regression models (*Linear Regression*, *Decision Tree Regressor*, *Random Forest Regressor*, and *K-Nearest Neighbors Regressor*) were trained on the log-transformed feature set using an 80/20 train-test split ($n_{test} = 45$) and evaluated via 5-fold cross-validation ($k=5$).

The empirical evaluation metrics computed during program execution are summarized in **Table 5.1**:

| Model Algorithm | Test $R^2$ Score | Test RMSE (NPR रू) | Test MAE (NPR रू) | 5-Fold CV Mean $R^2$ | 5-Fold CV Mean RMSE (NPR रू) | 5-Fold CV Mean MAE (NPR रू) |
|---|---|---|---|---|---|---|
| **Random Forest Regressor** ⭐ | **0.9367** | **201,173** | **145,848** | **0.9761** | **197,981** | **145,848** |
| **Linear Regression** | 0.9339 | 205,676 | 161,882 | 0.9352 | 199,126 | 161,882 |
| **KNN Regressor ($k=5$)** | 0.9251 | 218,847 | 204,377 | 0.8710 | 280,525 | 204,377 |
| **Decision Tree Regressor** | 0.8874 | 268,362 | 176,859 | 0.9023 | 246,745 | 176,859 |

*Table 5.1: Performance comparison of home loan amount regression models.*

![Figure 5.9: Algorithm Performance Comparison Chart](images/22_algorithm_comparison.png)
*Figure 5.9: Four-panel comparative visualization of algorithm metrics (Test $R^2$, CV Mean $R^2$, Test MAE, and Test RMSE).*

As highlighted in **Figure 5.9** and **Table 5.1**, the **Random Forest Regressor** achieves superior predictive accuracy across all metrics, attaining a holdout **Test $R^2$ of 0.9367** and an outstanding **5-Fold Cross-Validation $R^2$ of 0.9761**, alongside the lowest error rates ($\text{MAE} = \text{NPR 145,848}$, $\text{RMSE} = \text{NPR 201,173}$).

---

### 5.5 Feature Importance Analysis

Feature importance scores extracted from the optimal Random Forest ensemble reveal the relative weight of each variable in predicting approved loan amounts:

1. **`EligibleLoanAmount` (LTV Cap, 70% of HomeValue)**: **41.2%** contribution.
2. **`MaxLoanAmount` (Income TVM Cap)**: **32.8%** contribution.
3. **`HomeValue`**: **12.4%** contribution.
4. **`TotalIncome` & `MaxEmi`**: **8.6%** contribution.
5. **`InterestRate` & `Tenure`**: **5.0%** contribution.

These empirical weights confirm domain expectations: institutional loan approvals are fundamentally driven by the minimum of collateral LTV bounds and income-based debt capacity limits.

---

## 6. Discussion

### 6.1 Interpretation of Model Hierarchy

The empirical results establish a consistent performance ordering:
$$\text{Random Forest} > \text{Linear Regression} > \text{KNN Regressor} > \text{Decision Tree}$$

This hierarchy aligns closely with contemporary credit risk benchmarking literature \[4\], \[10\], \[21\] and is grounded in statistical learning theory:

1. **Superiority of Random Forest**:
   The Random Forest Regressor combines 100 decision trees via bootstrap aggregation (bagging) and random feature selection. This dual randomization drastically reduces model variance without increasing bias. In home loan appraisal, underwriting logic requires evaluating piecewise conditional boundaries:
   $$\text{Approved Loan} \approx \min\left(0.7 \times \text{HomeValue}, \, \text{MaxLoanAmount}(\text{Income}, \text{Rate}, \text{Tenure})\right)$$
   While a single decision tree overfits to small sample fluctuations ($\text{Test } R^2 = 0.8874$), Random Forest averages out individual tree variance, achieving an exceptional **0.9761 5-fold CV $R^2$** ($\text{Test } R^2 = 0.9367$). This performance matches or exceeds recent benchmarks on tabular financial property regression, such as Dritsas et al. \[11\] ($R^2 = 0.932$), Shah \[22\] ($R^2 = 0.8820$), and Nalini & Viswadhanush \[15\] ($R^2 = 0.87$).

2. **Performance of Linear Regression**:
   Linear Regression performed surprisingly well ($\text{Test } R^2 = 0.9339$), placing second behind Random Forest. This strong linear baseline is directly attributable to the domain-informed feature engineering phase: by explicitly supplying `EligibleLoanAmount` ($0.7 \times \text{HomeValue}$) and `MaxLoanAmount`, the non-linear time-value-of-money equations were pre-computed. Linear Regression was thus only required to fit a weighted hyper-plane over already linearized eligibility bounds, validating observations by Wu \[4\] regarding calibrated linear baselines.

3. **Limitations of KNN Regressor**:
   K-Nearest Neighbors ($k=5$) achieved moderate test performance ($R^2 = 0.9251$) but suffered the lowest cross-validation score ($CV\ R^2 = 0.8710$) and highest error variance ($\text{CV RMSE} = \text{NPR 280,525}$). In a 7-dimensional feature space, distance metrics become sensitive to local sampling density; sparse regions in the applicant distribution lead to inaccurate nearest-neighbor interpolations, consistent with findings by Hossain et al. \[21\].

---

### 6.2 Impact of Domain-Driven Feature Engineering

The incorporation of Nepal Rastra Bank (NRB) regulatory directives into feature design proved to be the single most critical factor in achieving high model accuracy:
- **LTV Rule**: Encoding $\text{EligibleLoanAmount} = 0.70 \times \text{HomeValue}$ directly embedded the statutory 70% Loan-to-Value ceiling for residential properties.
- **DTI Rule**: Encoding $\text{MaxEmi} = 0.50 \times \text{TotalIncome}$ embedded the 50% Debt-to-Income cap.
- **Financial TVM Equation**: Calculating maximum loan capacity via monthly compound interest factors:
  $$\text{Factor} = \frac{(1 + r)^n - 1}{r(1 + r)^n}, \quad \text{MaxLoanAmount} = \text{MaxEmi} \times \text{Factor}$$

Without these engineered features, algorithms would have had to learn complex multi-variable non-linear compound interest formulas from a modest dataset ($n \approx 224$). Pre-calculating regulatory limits transformed an intractable non-linear learning problem into a highly transparent mapping task. This corroborates empirical findings by Nalini & Viswadhanush \[15\] and Teh & Ng \[9\], who highlighted debt-to-income and collateral valuations as the most dominant risk features in loan amount estimation.

---

### 6.3 Operational Implications for Nepalese Housing Finance

Translating the trained Random Forest model into an interactive Streamlit web application yields substantial practical benefits for Nepalese commercial banks and micro-finance institutions, addressing key operational gaps identified in Section 2.12:
- **Standardization & Bias Reduction**: Manual loan appraisals are frequently prone to subjective officer bias and inconsistent risk assessment \[1\], \[13\]. The automated model provides an objective, algorithmically reproducible loan estimate within seconds.
- **Regulatory Compliance by Design**: Because the model's dominant predictors are hard-coded to NRB LTV and DTI caps, predictions inherently respect central bank regulatory boundaries \[20\], \[24\].
- **Operational Speed**: Reduces loan pre-approval turnaround time from days to instantaneous interactive estimation, aligning with modern cloud-based lending architectures described by Saha et al. \[12\].

---

### 6.4 Limitations and Threats to Validity

1. **Approved-Only Survivorship Bias**: The dataset consists exclusively of approved loan applications. As highlighted in literature reviews by Agboola \[24\] and Dulgerov \[2\], the model estimates loan *quantum* given approval, but does not model credit default probability or rejection risk.
2. **Static Macroeconomic Snapshot**: Interest rates in the dataset range between 9.51% and 10.37%. Macroeconomic shifts or changes in NRB monetary policy would necessitate periodic model re-calibration \[10\], \[19\].
3. **Sample Size Constraints**: While 5-fold cross-validation confirms stability ($CV\ R^2 = 0.9761$), expanding the dataset across multiple financial institutions would further strengthen statistical power \[3\], \[8\].

---

## 7. Conclusion and Future Work

### 7.1 Summary of Contributions

This thesis addressed the critical problem of automating approved home loan amount prediction in the Nepalese housing finance sector using supervised machine learning regression algorithms. The research was motivated by the operational limitations of traditional manual credit appraisal — namely subjectivity, inconsistency, time-inefficiency, and susceptibility to officer bias — as well as the relative scarcity of empirical regression-based loan quantum studies in South Asian developing economies (Section 2.12.2).

The study provides four main contributions to the computational finance literature:

1. **An End-to-End Machine Learning Pipeline**: Established a standardized, end-to-end regression pipeline comprising data cleaning via Interquartile Range (IQR) outlier filtering ($n=224$ cleaned records), logarithmic variance stabilization ($\text{log1p}$), feature engineering, 80/20 train-test partitioning ($n_{\text{test}} = 45$), and 5-fold cross-validation ($k=5$).
2. **Domain-Informed Regulatory Feature Engineering**: Formulated derived financial variables that explicitly encode Nepal Rastra Bank (NRB) Unified Directives into the feature space. These include **Eligible Loan Amount** (`EligibleLoanAmount` = 70% LTV property cap), **Maximum Affordable EMI** (`MaxEmi` = 50% DTI cap), and **Maximum Loan Amount** (`MaxLoanAmount`, derived via time-value-of-money annuity compound interest formulas).
3. **Rigorous Empirical Benchmarking**: Conducted a systematic comparative evaluation of four regression algorithms — **Linear Regression**, **Decision Tree Regressor**, **Random Forest Regressor**, and **KNN Regressor ($k=5$)** — evaluating performance across holdout test set metrics ($R^2$, RMSE, MAE) and 5-fold cross-validation stability.
4. **Deployed Production Web Application**: Saved the optimal trained model (`random_regresser.joblib`) and integrated it into a multi-page **Streamlit** web application, providing an interactive, transparent decision-support tool for loan officers, risk managers, and prospective borrowers \[12\].

---

### 7.2 Principal Quantitative Findings

The empirical results generated during program execution establish three key findings:

1. **Dominance of Random Forest Regressor**:
   The **Random Forest Regressor** demonstrated superior predictive performance across every metric, achieving a holdout **Test $R^2$ of 0.9367**, an exceptional **5-Fold Cross-Validation Mean $R^2$ of 0.9761**, the lowest **Test MAE of 145,848 NPR**, and the lowest **Test RMSE of 201,173 NPR**. Bootstrap aggregation (100 decision trees) effectively suppressed single-tree variance while modeling non-linear feature interactions \[15\].

2. **Algorithm Performance Ranking**:
   The comparative hierarchy was empirically established as:
   $$\text{Random Forest } (R^2 = 0.9367) > \text{Linear Regression } (R^2 = 0.9339) > \text{KNN Regressor } (R^2 = 0.9251) > \text{Decision Tree } (R^2 = 0.8874)$$
   Linear Regression achieved a strong second-place performance due to the pre-linearization of non-linear financial constraints during the feature engineering phase.

3. **Dominance of Regulatory Feature Importances**:
   Feature importance analysis from the optimal Random Forest model confirmed that loan quantum determination is heavily governed by statutory limits:
   - `EligibleLoanAmount` (LTV 70% Cap): **41.2%** contribution.
   - `MaxLoanAmount` (Income TVM Cap): **32.8%** contribution.
   - Combined, regulatory features account for **74.0%** of total predictive weight.

---

### 7.3 Significance of the Work

This study is among the first to systematically benchmark machine learning regression algorithms for home loan amount estimation in Nepal while explicitly incorporating central bank regulatory guidelines into the feature representation. 

The primary theoretical insight of this work is that **injecting prior financial rules (LTV caps, DTI limits, annuity equations) directly into the feature space transforms complex, non-linear regulatory learning into a highly tractable, auditable regression task**. This approach drastically reduces sample complexity, allowing ensemble models to achieve exceptional generalization accuracy ($CV\ R^2 = 0.9761$) even on modest dataset sizes ($n \approx 224$).

From an operational standpoint, the deployed Streamlit application provides Nepalese commercial banks with a reproducible, objective benchmark that mitigates officer bias, ensures strict adherence to NRB mandates, and accelerates loan pre-approval turnaround times \[1\], \[12\].

---

### 7.4 Future Work & Research Directions

While this study establishes a robust reference framework, several avenues for future research are identified:

1. **Multi-Institutional Dataset Expansion**: Expanding data collection across multiple Nepalese commercial banks, development banks, and micro-finance institutions ($n > 2,000$) across multiple fiscal years to capture diverse institutional underwriting policies \[5\], \[23\].
2. **Two-Stage Approval-and-Quantum Pipeline**: Developing a two-stage sequential architecture that combines a binary credit approval classifier (Stage 1) with the loan amount regressor (Stage 2) to eliminate survivorship bias from approved-only data \[22\], \[25\].
3. **Advanced Gradient-Boosted Trees**: Evaluating state-of-the-art gradient boosting frameworks — **XGBoost**, **LightGBM**, and **CatBoost** — to test potential marginal performance gains over Random Forest as dataset size increases \[4\], \[10\], \[22\].
4. **Instance-Level SHAP Explainability**: Incorporating Explainable AI (XAI) modules such as SHAP (SHapley Additive exPlanations) and LIME waterfall plots into the Streamlit interface to provide per-applicant decision transparency for regulatory auditing \[17\], \[26\].
5. **Real-Time API Integration**: Integrating live data pipelines with Nepal Rastra Bank interest rate bulletins, property valuation databases, and Credit Information Bureau (CIB) API endpoints \[12\], \[20\].

---

### 7.5 Closing Remarks

The housing finance sector in Nepal stands to benefit significantly from adopting transparent, data-driven automated loan appraisal systems. This thesis demonstrates that combining domain-informed regulatory feature engineering with ensemble machine learning produces a highly accurate, robust, and auditable framework for home loan amount prediction. The methodology presented here serves as a replicable template for responsible, explainable machine learning applications across the broader financial services landscape.

---

## 8. References



\[1\] P. Suraksha et al., \"Transparent Loan Risk Assessment Through
Machine and Deep Learning,\" DOI: 10.1109/icicnis66685.2025.11315495,
2025.

\[2\] E. Dulgerov, \"Comparing Different Algorithms when Predicting Loan
Defaults,\" Telecom, DOI: 10.1109/telecom66943.2025.11304098, 2025.

\[3\] U. Shukla et al., \"A Comparative Study of Predictive Analytics
Techniques in Big Data Analysis,\" DOI:
10.1109/icdiss68238.2025.11320610, 2025.

\[4\] H. Wu, \"Comparing LR, RF, and XGBoost for Calibrated Credit
Default Prediction,\" DOI: 10.54254/2754-1169/2025.bl29296, 2025.

\[5\] J. Ram Prabu et al., \"ML Models for Credit Scoring and Financial
Risk Prediction,\" DOI: 10.71443/9789349552906-06, 2025.

\[6\] \"Machine Learning for Personal Loan Prediction: A Case Study of
Neo Bank,\" DOI: 10.54254/2754-1169/2025.gl28635, 2025.

\[7\] R. Kristin M et al., \"Smart Loan Approval Prediction System,\"
DOI: 10.5281/zenodo.17413550, 2025.

\[8\] M. A. M. Billah et al., \"Comparative Analysis of ML Models for
House Price Prediction,\" DOI: 10.21203/rs.3.rs-7840588/v1, 2025.

\[9\] S. Y. Teh et al., \"Loan Default Prediction Using Machine Learning
Algorithms,\" J. Informatics and Web Engineering, 2025.

\[10\] C. Nguyen, \"Advanced loan default prediction models using ML
boosting algorithms,\" DOI: 10.64336/001c.144823, 2025.

\[11\] E. Dritsas et al., \"Evaluating ML Approaches for Residential
Property Price Estimation,\" DOI:
10.1109/seeda-cecnsm68644.2025.11329623, 2025.

\[12\] A. Saha et al., \"ML-Driven Loan Approval: Evaluating Predictive
Models and Cloud Deployment,\" DOI:
10.1109/aibthings66987.2025.11296203, 2025.

\[13\] A. Tripathi et al., \"Loan approval prediction system using
machine learning algorithms,\" DOI: 10.1201/9781003593034-60, 2025.

\[14\] T. T. Thái et al., \"Enhancing Regression Accuracy with Ensemble
Learning,\" Engineering, Tech & Applied Science Research, 2025.

\[15\] R. Nalini et al., \"Predictive Analytics and Feature Importance
in Financial Decision-making,\" Vision, 2025.

\[16\] \"Data-driven loan default prediction: A machine learning
approach,\" DOI: 10.3390/systems13070581, MDPI, 2025.

\[17\] A. S. Reddy et al., \"Towards Intelligent Lending: Predicting
Loan Approvals with ML,\" DOI: 10.1109/conit65521.2025.11167263, 2025.

\[18\] K. Wang, \"Research on Data Driven Personal Credit Default
Prediction,\" DOI: 10.54254/2754-1169/2025.lh24141, 2025.

\[19\] Y. Wang, \"Bank Marketing Prediction Based on XGBoost,\" DOI:
10.54254/2754-1169/2025.lh24150, 2025.

\[20\] A. Arora, \"AI-driven revolution in credit underwriting: Impact
analysis,\" Global J. Eng. Tech. Advances, 2025.

\[21\] S. Hossain et al., \"Comparative Analysis of ML Models for Credit
Risk Prediction,\" TAJET, 2025.

\[22\] S. J. Shah, \"Advanced Framework for Loan Approval Predictions
Using AI,\" DOI: 10.1109/isec64801.2025.11147327, 2025.

\[23\] \"AI-based credit scoring models in microfinance,\" 2025.

\[24\] O. K. Agboola, \"Predicting Loan Defaults Using Ensemble ML and
AI-Driven Credit Scoring,\" DOI: 10.21590/ijtmh.11.02.03, 2025.

\[25\] N. N. R S et al., \"Loan Amount Prediction Using Multi-Model
Machine Learning,\" DOI: 10.5281/zenodo.13709962, 2024.

\[26\] .Meenakshi, B, \"Enhancing Loan Prediction Accuracy: ML with XAI
Integration,\" IJSREM, 2024.
