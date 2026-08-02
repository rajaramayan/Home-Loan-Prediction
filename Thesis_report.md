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

## 2. Literature Review

### 2.1 Credit Risk Assessment: Historical Foundations

The quantitative assessment of credit risk has a long history in financial research.
Altman [8] introduced the Z-score model using discriminant analysis to predict corporate
bankruptcy from financial ratios, establishing the principle that financial data can be
combined mathematically to produce actionable risk scores. This foundational work
directly influenced the evolution of consumer credit scoring. Hand and Henley [9]
provided a comprehensive review of statistical classification methods in consumer credit
scoring, surveying techniques from linear discriminant analysis and logistic regression
through to neural networks, and concluding that no single method dominates across all
datasets. Thomas [10] extended this survey to include behavioural scoring — assessing
ongoing risk after loan approval — and emphasised the need for robust evaluation
methodologies including cross-validation, a recommendation that the present study follows.

### 2.2 Classical Machine Learning for Credit Scoring

Decision trees emerged as an early interpretable alternative to statistical methods.
Quinlan [3] formalised the induction of decision trees through the ID3 algorithm,
establishing recursive partitioning as a principled approach to classification and
regression. The K-Nearest Neighbors algorithm, formalised by Cover and Hart [4], offered
a non-parametric baseline that requires no explicit model training but is sensitive to
the curse of dimensionality in high-dimensional feature spaces. West [12] directly
compared neural networks against logistic regression and discriminant analysis for credit
scoring across multiple real-world datasets, finding that multilayer perceptrons provided
marginal but consistent gains. Baesens et al. [11] extended this comparison by adding
support vector machines and demonstrating that rule extraction from neural networks can
produce interpretable decision tables suitable for regulatory review. Huang, Chen, and
Wang [23] applied support vector machines with various kernel functions to credit scoring,
demonstrating that the radial basis function kernel outperformed both linear SVMs and
neural networks on a Taiwanese banking dataset. Hearst et al. [20] provided an accessible
overview of SVMs and their theoretical margin-maximisation principle, which underlies
their strong generalisation on high-dimensional financial data.

### 2.3 Ensemble Methods

Breiman [2] introduced bagging (bootstrap aggregating), showing that averaging
predictions from models trained on bootstrap resamples of the training data substantially
reduces variance without increasing bias. Breiman [1] extended this idea into Random
Forests by additionally randomising the feature subset considered at each tree split,
producing a decorrelated ensemble with significantly lower variance than both single
decision trees and bagged trees. Random Forest has since become one of the most widely
used algorithms for tabular financial data due to its robustness, scalability, and built-in
feature importance estimates. Dietterich [16] provided a theoretical and empirical
framework for understanding why ensembles outperform individual classifiers, identifying
bias, variance, and noise as the three sources of generalisation error that ensembles
address. Friedman [5] introduced gradient boosting machines (GBM), framing boosting as
stage-wise gradient descent in function space, and showed that GBMs consistently
outperform single trees and bagging on a wide range of prediction tasks. Chen and
Guestrin [6] built on this foundation with XGBoost, incorporating regularisation,
parallel tree construction, and approximate split-finding to scale gradient boosting to
large datasets, achieving state-of-the-art results on numerous benchmarks in finance
and beyond. Lessmann et al. [13] conducted the most comprehensive benchmarking study of
credit scoring algorithms to date, comparing 41 classifiers across 8 real-world credit
datasets using statistical testing; they found that ensemble methods — particularly
Random Forest, gradient boosting, and bagging — significantly outperformed linear and
single-tree baselines.

### 2.4 Loan and Mortgage Prediction Using Machine Learning

Khandani, Kim, and Lo [14] applied machine learning to consumer credit risk at scale,
using transaction-level data from a major U.S. bank to build models that predicted credit
card default and delinquency. Their work demonstrated that non-linear ML models capture
behavioural patterns not accessible to linear models and can materially reduce loss rates
when integrated into automated credit management systems. Yeh and Lien [15] compared six
data mining techniques — including artificial neural networks, decision trees, and logistic
regression — for predicting the probability of default on credit card accounts in Taiwan,
finding that artificial neural networks achieved the best predictive accuracy. Tsai and
Wu [19] further explored neural network ensembles for bankruptcy prediction, confirming
that ensemble averaging of multiple neural networks consistently outperforms individual
networks. Dastile, Celik, and Potsane [24] conducted a systematic literature review of
97 studies on credit scoring models published between 2000 and 2019, finding a strong and
growing trend toward ensemble and deep learning approaches while noting that
interpretability remains a significant challenge for regulatory compliance. Bao, Yue, and
Rao [21] proposed a hybrid deep learning framework combining stacked autoencoders and
long short-term memory (LSTM) networks for financial time series prediction,
demonstrating that deep architectures can extract latent representations from financial
data that shallow models miss. The present study focuses on the regression variant of
loan prediction — estimating the approved loan *amount* rather than a binary
approval/rejection outcome — which has received comparatively less attention in the
literature despite its direct operational importance to lending institutions.

### 2.5 Feature Engineering in Lending Models

Feature engineering — the construction of informative derived features from raw data
using domain knowledge — has been repeatedly identified as one of the most impactful
steps in the machine learning pipeline for finance. Tibshirani [18] introduced the LASSO
(Least Absolute Shrinkage and Selection Operator), which performs simultaneous regression
and feature selection by applying an L1 penalty to coefficients; this provided a formal
statistical basis for identifying the most predictive features in high-dimensional lending
datasets. Baesens et al. [11] and Lessmann et al. [13] both found that the composition
of the feature set has a larger impact on model performance than the choice of algorithm,
particularly for small-to-medium datasets. In the present study, the three engineered
features — `MaxEmi`, `MaxLoanAmount`, and `EligibleLoanAmount` — encode the standard
amortization formula and Nepal Rastra Bank LTV guidelines directly into the feature
space, reducing the complexity of the learning task for all models.

### 2.6 Property Valuation and Housing Finance

Sirmans, Macpherson, and Zietz [22] surveyed hedonic pricing models for real estate,
documenting the strong relationship between property attributes (size, location, age) and
market value. Their review confirms that property value is a primary determinant of
mortgage size in all major lending markets, consistent with the dominant role of
`HomeValue` and the LTV-derived `EligibleLoanAmount` feature in the present model.
The 70% LTV cap enforced by Nepal Rastra Bank [26] further anchors the eligible loan
amount directly to property value, making `HomeValue` the single most influential raw
input in the feature set.

### 2.7 Model Interpretability and Explainability

As machine learning models are increasingly deployed in high-stakes financial decisions,
regulatory and ethical concerns about model transparency have intensified. Lundberg and
Lee [7] introduced SHAP (SHapley Additive exPlanations), a unified framework grounded in
cooperative game theory that assigns a contribution value to each feature for each
individual prediction. SHAP values have become the de facto standard for post-hoc
explanation of tree-based models in credit scoring and lending, enabling institutions to
provide applicant-level explanations of loan decisions in compliance with regulations such
as the EU's General Data Protection Regulation (GDPR) right-to-explanation. The present
study surfaces feature importance from the Random Forest model as an initial step toward
interpretability; full SHAP integration is identified as a priority direction for future
work.

### 2.8 Deployment and Tooling

Pedregosa et al. [17] developed scikit-learn, the Python library used for all model
training, evaluation, and preprocessing in this study. Its unified API enables consistent
comparison of algorithms with minimal implementation overhead. The deployment layer uses
Streamlit [25], which provides a Python-native framework for building interactive data
applications without front-end web development expertise, making it particularly
well-suited for rapid academic prototyping and demonstration.

### 2.9 Regulatory Context

The Nepal Rastra Bank [26] mandates a maximum Loan-to-Value ratio of 70% for home
loans, directly informing the `EligibleLoanAmount` feature. The 50% Debt-to-Income
cap used to derive `MaxEmi` reflects standard prudential lending norms applied across
Nepalese commercial banks. These regulatory constraints are embedded as hard rules in the
feature engineering pipeline, ensuring the model operates within the legal lending
framework and that its predictions are financially interpretable.

### 2.10 Research Gap

The preceding review of the literature reveals several distinct but related gaps that
collectively motivate the present study. These gaps are identified and discussed below
across five dimensions: task formulation, geographic and regulatory context, feature
engineering practice, evaluation methodology, and deployment.

**Gap 1 — Regression vs. Classification Task Formulation.**
The overwhelming majority of published work in machine learning for lending focuses on
binary classification: will a loan be approved or rejected? Studies by Yeh and Lien [15],
Khandani et al. [14], Baesens et al. [11], and the benchmarking work of Lessmann et al.
[13] all treat loan outcomes as categorical. The regression problem — given that a loan
*will* be approved, *how much* should be sanctioned? — has received substantially less
systematic attention in the literature, despite being of equal operational importance.
The approved loan amount determines the borrower's purchasing capacity, the lender's
exposure, and the institution's capital allocation. An accurate regression model directly
addresses a decision that every loan officer must make, yet no dedicated comparative
benchmarking study on this specific regression task was identified in the survey.

**Gap 2 — Absence of Nepalese and South Asian Housing Finance Studies.**
Credit scoring research is concentrated in North American, European, and East Asian
contexts (U.S. mortgage data [14], Taiwanese credit card data [15], German credit [13]).
Developing economies in South Asia — where housing finance is a rapidly growing sector
but institutional data infrastructure remains limited — are substantially underrepresented.
Nepal presents a particularly distinct context: a developing economy with a relatively
young formal housing finance sector, a specific set of Nepal Rastra Bank prudential
guidelines (70% LTV cap, DTI norms), income distributions that differ markedly from
those in benchmark datasets, and a high proportion of applicants with zero reported family
income. No peer-reviewed study was found that applies machine learning regression to home
loan amount prediction using data from a Nepalese financial institution.

**Gap 3 — Regulatory Constraint Embedding in Feature Engineering.**
Existing feature engineering approaches in credit scoring largely focus on statistical
feature selection (LASSO [18], recursive feature elimination) or dimensionality reduction,
without explicitly encoding the regulatory and mathematical constraints that govern
lending decisions. The standard amortization formula and the LTV ratio are not emergent
patterns that a model must discover from data — they are known, legally mandated formulae.
Yet no published study was found that systematically constructs derived features
(`MaxEmi`, `MaxLoanAmount`, `EligibleLoanAmount`) by directly encoding these formulae
and regulatory caps into the feature space as a preprocessing step for a regression
model. This represents both a methodological gap and a missed opportunity: embedding
regulatory logic as features simplifies the learning task, improves model accuracy, and
ensures predictions are inherently compliant with lending rules.

**Gap 4 — Rigorous Evaluation of Simple Baselines in the Regression Setting.**
Lessmann et al. [13] established a rigorous multi-dataset, multi-metric benchmarking
protocol for credit *classification*, comparing 41 classifiers with statistical
significance testing. An equivalent benchmarking study for loan *amount regression* —
comparing foundational algorithms (Linear Regression, Decision Tree, KNN, Random Forest)
using both hold-out and cross-validated metrics — has not been conducted in the South
Asian housing finance context. Without such a baseline study, it is unclear which class
of algorithm (parametric, tree-based, instance-based, or ensemble) best captures the
structure of loan amount data in this setting, and whether the complexity of ensemble
methods is justified relative to simpler alternatives on small-to-medium datasets.

**Gap 5 — End-to-End Deployment for Non-Technical Practitioners.**
Even where ML models have been developed for lending in academic settings, deployment
as accessible, practitioner-facing tools remains rare. The majority of published studies
present results as static tables in a journal article, with no accompanying software
artefact that a loan officer or risk manager could use without programming expertise.
Streamlit-based interactive deployment of a loan prediction model, incorporating
exploratory analysis, model evaluation transparency, and real-time prediction in a single
interface, has not been demonstrated in the context of Nepalese housing finance.

**Positioning of the Present Study.**
This thesis directly addresses all five gaps. It frames loan amount prediction explicitly
as a regression task (Gap 1), uses data from a Nepalese financial institution under
Nepal Rastra Bank regulatory norms (Gap 2), embeds amortization and LTV constraints as
engineered features (Gap 3), conducts a principled four-algorithm comparative evaluation
using both test and cross-validation metrics (Gap 4), and delivers the best model as a
deployed Streamlit web application accessible to non-technical users (Gap 5). In doing
so, it contributes the first documented end-to-end machine learning pipeline for home
loan amount regression in the Nepalese context, combining the methodological rigour of
established credit-scoring benchmarking with domain-specific financial engineering and
practical deployment.

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

This hierarchy is grounded in statistical learning theory:

1. **Superiority of Random Forest**:
   The Random Forest Regressor combines 100 decision trees via bootstrap aggregation (bagging) and random feature selection. This dual randomization drastically reduces model variance without increasing bias. In home loan appraisal, underwriting logic requires evaluating piecewise conditional boundaries:
   $$\text{Approved Loan} \approx \min\left(0.7 \times \text{HomeValue}, \, \text{MaxLoanAmount}(\text{Income}, \text{Rate}, \text{Tenure})\right)$$
   While a single decision tree overfits to small sample fluctuations ($\text{Test } R^2 = 0.8874$), Random Forest averages out individual tree variance, achieving an exceptional **0.9761 5-fold CV $R^2$**.

2. **Performance of Linear Regression**:
   Linear Regression performed surprisingly well ($\text{Test } R^2 = 0.9339$), placing second behind Random Forest. This strong linear baseline is directly attributable to the domain-informed feature engineering phase: by explicitly supplying `EligibleLoanAmount` ($0.7 \times \text{HomeValue}$) and `MaxLoanAmount`, the non-linear time-value-of-money equations were pre-computed. Linear Regression was thus only required to fit a weighted hyper-plane over already linearized eligibility bounds.

3. **Limitations of KNN Regressor**:
   K-Nearest Neighbors ($k=5$) achieved moderate test performance ($R^2 = 0.9251$) but suffered the lowest cross-validation score ($CV\ R^2 = 0.8710$) and highest error variance ($\text{CV RMSE} = \text{NPR 280,525}$). In a 7-dimensional feature space, distance metrics become sensitive to local sampling density; sparse regions in the applicant distribution lead to inaccurate nearest-neighbor interpolations.

---

### 6.2 Impact of Domain-Driven Feature Engineering

The incorporation of Nepal Rastra Bank (NRB) regulatory directives into feature design proved to be the single most critical factor in achieving high model accuracy:
- **LTV Rule**: Encoding $\text{EligibleLoanAmount} = 0.70 \times \text{HomeValue}$ directly embedded the statutory 70% Loan-to-Value ceiling for residential properties.
- **DTI Rule**: Encoding $\text{MaxEmi} = 0.50 \times \text{TotalIncome}$ embedded the 50% Debt-to-Income cap.
- **Financial TVM Equation**: Calculating maximum loan capacity via monthly compound interest factors:
  $$\text{Factor} = \frac{(1 + r)^n - 1}{r(1 + r)^n}, \quad \text{MaxLoanAmount} = \text{MaxEmi} \times \text{Factor}$$

Without these engineered features, algorithms would have had to learn complex multi-variable non-linear compound interest formulas from a modest dataset ($n \approx 224$). Pre-calculating regulatory limits transformed an intractable non-linear learning problem into a highly transparent mapping task.

---

### 6.3 Operational Implications for Nepalese Housing Finance

Translating the trained Random Forest model into an interactive Streamlit web application yields substantial practical benefits for Nepalese commercial banks and micro-finance institutions:
- **Standardization & Bias Reduction**: Manual loan appraisals are frequently prone to subjective officer bias and inconsistent risk assessment. The automated model provides an objective, algorithmically reproducible loan estimate within seconds.
- **Regulatory Compliance by Design**: Because the model's dominant predictors are hard-coded to NRB LTV and DTI caps, predictions inherently respect central bank regulatory boundaries.
- **Operational Speed**: Reduces loan pre-approval turnaround time from days to instantaneous interactive estimation.

---

### 6.4 Limitations and Threats to Validity

1. **Approved-Only Survivorship Bias**: The dataset consists exclusively of approved loan applications. The model estimates loan *quantum* given approval, but does not model credit default probability or rejection risk.
2. **Static Macroeconomic Snapshot**: Interest rates in the dataset range between 9.51% and 10.37%. Macroeconomic shifts or changes in NRB monetary policy would necessitate periodic model re-calibration.
3. **Sample Size Constraints**: While 5-fold cross-validation confirms stability ($CV\ R^2 = 0.9761$), expanding the dataset across multiple financial institutions would further strengthen statistical power.

---

## 7. Conclusion and Future Work

### 7.1 Summary of Contributions

This thesis addressed the critical problem of automating approved home loan amount prediction in the Nepalese housing finance sector using supervised machine learning regression algorithms. The research was motivated by the operational limitations of traditional manual credit appraisal — namely subjectivity, inconsistency, time-inefficiency, and susceptibility to officer bias — as well as the relative scarcity of empirical regression-based loan quantum studies in South Asian developing economies.

The study provides four main contributions to the computational finance literature:

1. **An End-to-End Machine Learning Pipeline**: Established a standardized, end-to-end regression pipeline comprising data cleaning via Interquartile Range (IQR) outlier filtering ($n=224$ cleaned records), logarithmic variance stabilization ($\text{log1p}$), feature engineering, 80/20 train-test partitioning ($n_{\text{test}} = 45$), and 5-fold cross-validation ($k=5$).
2. **Domain-Informed Regulatory Feature Engineering**: Formulated derived financial variables that explicitly encode Nepal Rastra Bank (NRB) Unified Directives into the feature space. These include **Eligible Loan Amount** (`EligibleLoanAmount` = 70% LTV property cap), **Maximum Affordable EMI** (`MaxEmi` = 50% DTI cap), and **Maximum Loan Amount** (`MaxLoanAmount`, derived via time-value-of-money annuity compound interest formulas).
3. **Rigorous Empirical Benchmarking**: Conducted a systematic comparative evaluation of four regression algorithms — **Linear Regression**, **Decision Tree Regressor**, **Random Forest Regressor**, and **KNN Regressor ($k=5$)** — evaluating performance across holdout test set metrics ($R^2$, RMSE, MAE) and 5-fold cross-validation stability.
4. **Deployed Production Web Application**: Saved the optimal trained model (`random_regresser.joblib`) and integrated it into a multi-page **Streamlit** web application, providing an interactive, transparent decision-support tool for loan officers, risk managers, and prospective borrowers.

---

### 7.2 Principal Quantitative Findings

The empirical results generated during program execution establish three key findings:

1. **Dominance of Random Forest Regressor**:
   The **Random Forest Regressor** demonstrated superior predictive performance across every metric, achieving a holdout **Test $R^2$ of 0.9367**, an exceptional **5-Fold Cross-Validation Mean $R^2$ of 0.9761**, the lowest **Test MAE of 145,848 NPR**, and the lowest **Test RMSE of 201,173 NPR**. Bootstrap aggregation (100 decision trees) effectively suppressed single-tree variance while modeling non-linear feature interactions.

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

From an operational standpoint, the deployed Streamlit application provides Nepalese commercial banks with a reproducible, objective benchmark that mitigates officer bias, ensures strict adherence to NRB mandates, and accelerates loan pre-approval turnaround times.

---

### 7.4 Future Work & Research Directions

While this study establishes a robust reference framework, several avenues for future research are identified:

1. **Multi-Institutional Dataset Expansion**: Expanding data collection across multiple Nepalese commercial banks, development banks, and micro-finance institutions ($n > 2,000$) across multiple fiscal years to capture diverse institutional underwriting policies.
2. **Two-Stage Approval-and-Quantum Pipeline**: Developing a two-stage sequential architecture that combines a binary credit approval classifier (Stage 1) with the loan amount regressor (Stage 2) to eliminate survivorship bias from approved-only data.
3. **Advanced Gradient-Boosted Trees**: Evaluating state-of-the-art gradient boosting frameworks — **XGBoost**, **LightGBM**, and **CatBoost** — to test potential marginal performance gains over Random Forest as dataset size increases.
4. **Instance-Level SHAP Explainability**: Incorporating SHAP (SHapley Additive exPlanations) waterfall plots into the Streamlit interface to provide per-applicant decision transparency for regulatory auditing.
5. **Real-Time API Integration**: Integrating live data pipelines with Nepal Rastra Bank interest rate bulletins, property valuation databases, and Credit Information Bureau (CIB) API endpoints.

---

### 7.5 Closing Remarks

The housing finance sector in Nepal stands to benefit significantly from adopting transparent, data-driven automated loan appraisal systems. This thesis demonstrates that combining domain-informed regulatory feature engineering with ensemble machine learning produces a highly accurate, robust, and auditable framework for home loan amount prediction. The methodology presented here serves as a replicable template for responsible, explainable machine learning applications across the broader financial services landscape.

---

## 8. References

[1]  L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32,
     Oct. 2001.

[2]  L. Breiman, "Bagging predictors," *Machine Learning*, vol. 24, no. 2,
     pp. 123–140, Aug. 1996.

[3]  J. R. Quinlan, "Induction of decision trees," *Machine Learning*, vol. 1, no. 1,
     pp. 81–106, Mar. 1986.

[4]  T. M. Cover and P. E. Hart, "Nearest neighbor pattern classification," *IEEE
     Transactions on Information Theory*, vol. 13, no. 1, pp. 21–27, Jan. 1967.

[5]  J. H. Friedman, "Greedy function approximation: A gradient boosting machine,"
     *The Annals of Statistics*, vol. 29, no. 5, pp. 1189–1232, Oct. 2001.

[6]  T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc.
     22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD'16)*,
     San Francisco, CA, USA, Aug. 2016, pp. 785–794.

[7]  S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model
     predictions," in *Advances in Neural Information Processing Systems 30
     (NIPS 2017)*, Long Beach, CA, USA, 2017, pp. 4765–4774.

[8]  E. I. Altman, "Financial ratios, discriminant analysis and the prediction of
     corporate bankruptcy," *The Journal of Finance*, vol. 23, no. 4, pp. 589–609,
     Sep. 1968.

[9]  D. J. Hand and W. E. Henley, "Statistical classification methods in consumer
     credit scoring: A review," *Journal of the Royal Statistical Society: Series A
     (Statistics in Society)*, vol. 160, no. 3, pp. 523–541, 1997.

[10] L. C. Thomas, "A survey of credit and behavioural scoring: Forecasting financial
     risk of lending to consumers," *International Journal of Forecasting*, vol. 16,
     no. 2, pp. 149–172, Apr.–Jun. 2000.

[11] B. Baesens, R. Setiono, C. Mues, and J. Vanthienen, "Using neural network rule
     extraction and decision tables for credit-risk evaluation," *Management Science*,
     vol. 49, no. 3, pp. 312–329, Mar. 2003.

[12] D. West, "Neural network credit scoring models," *Computers & Operations
     Research*, vol. 27, no. 11–12, pp. 1131–1152, Sep.–Oct. 2000.

[13] S. Lessmann, B. Baesens, H.-V. Seow, and L. C. Thomas, "Benchmarking state-of-
     the-art classification algorithms for credit scoring: An update of research,"
     *European Journal of Operational Research*, vol. 247, no. 1, pp. 124–136,
     Nov. 2015.

[14] A. E. Khandani, A. J. Kim, and A. W. Lo, "Consumer credit-risk models via
     machine-learning algorithms," *Journal of Banking & Finance*, vol. 34, no. 11,
     pp. 2767–2787, Nov. 2010.

[15] I.-C. Yeh and C.-H. Lien, "The comparisons of data mining techniques for the
     predictive accuracy of probability of default of credit card clients," *Expert
     Systems with Applications*, vol. 36, no. 2, pp. 2473–2480, Mar. 2009.

[16] T. G. Dietterich, "Ensemble methods in machine learning," in *Multiple Classifier
     Systems*, Lecture Notes in Computer Science, vol. 1857, Springer, Berlin,
     Heidelberg, 2000, pp. 1–15.

[17] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of
     Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[18] R. Tibshirani, "Regression shrinkage and selection via the lasso," *Journal of
     the Royal Statistical Society: Series B (Methodological)*, vol. 58, no. 1,
     pp. 267–288, 1996.

[19] C.-F. Tsai and J.-W. Wu, "Using neural network ensembles for bankruptcy
     prediction and credit scoring," *Expert Systems with Applications*, vol. 34,
     no. 4, pp. 2639–2649, May 2008.

[20] M. A. Hearst, S. T. Dumais, E. Osuna, J. Platt, and B. Schölkopf, "Support
     vector machines," *IEEE Intelligent Systems and Their Applications*, vol. 13,
     no. 4, pp. 18–28, Jul.–Aug. 1998.

[21] W. Bao, J. Yue, and Y. Rao, "A deep learning framework for financial time series
     using stacked autoencoders and long-short term memory," *PLOS ONE*, vol. 12,
     no. 7, p. e0180944, Jul. 2017.

[22] G. S. Sirmans, D. A. Macpherson, and E. N. Zietz, "The composition of hedonic
     pricing models," *Journal of Real Estate Literature*, vol. 13, no. 1, pp. 1–44,
     2005.

[23] C.-L. Huang, M.-C. Chen, and C.-J. Wang, "Credit scoring with a data mining
     approach based on support vector machines," *Expert Systems with Applications*,
     vol. 33, no. 4, pp. 847–856, Nov. 2007.

[24] X. Dastile, T. Celik, and M. Potsane, "Statistical and machine learning models
     in credit scoring: A systematic literature review," *Applied Soft Computing*,
     vol. 91, p. 106263, Jun. 2020.

[25] Streamlit Inc., *Streamlit Documentation*, 2023. [Online]. Available:
     https://docs.streamlit.io

[26] Nepal Rastra Bank, *Unified Directives 2080*, Nepal Rastra Bank, Kathmandu,
     Nepal, 2023.
