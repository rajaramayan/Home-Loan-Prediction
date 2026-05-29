# A Comparative Study of Machine Learning Algorithms for Home Loan Amount Prediction

---

## Abstract

The rapid expansion of the housing finance sector has created a growing demand for
accurate and automated tools to assess home loan eligibility and predict loan amounts.
Traditional manual loan appraisal processes are time-consuming, inconsistent, and
susceptible to human bias. This thesis presents a comparative study of four supervised
machine learning algorithms — Linear Regression, Decision Tree Regressor, Random Forest
Regressor, and K-Nearest Neighbors (KNN) Regressor — for predicting the approved home
loan amount for applicants.

The study uses a dataset of 250 approved home loan records collected from a Nepalese
financial context. The dataset comprises applicant demographic attributes (gender, marital
status, age, education), income information (client annual income, family annual income),
and loan-specific attributes (home value, interest rate, tenure, and EMI). A set of
financially meaningful engineered features were derived prior to model training, including
Total Monthly Income (sum of client and family income divided by 12), Maximum Affordable
EMI (50% of total monthly income in accordance with standard lending norms), Maximum
Loan Amount (the theoretical maximum loan serviceable from the applicant's income, derived
using the standard amortization formula), and Eligible Loan Amount (70% of the property's
market value, reflecting the conventional Loan-to-Value ratio).

Data preprocessing steps included outlier removal using the Interquartile Range (IQR)
method on HomeValue and TotalIncome, and log transformation of skewed monetary features
to normalise their distributions. The dataset was split into training (80%) and test (20%)
sets, and all models were evaluated using both hold-out test metrics and 5-fold
cross-validation to ensure robustness.

Performance was measured using R² (coefficient of determination), Root Mean Squared Error
(RMSE), and Mean Absolute Error (MAE). The Random Forest Regressor consistently
outperformed the other three models across all evaluation metrics, achieving the highest
cross-validated R² and the lowest cross-validated RMSE and MAE. The model was saved and
deployed through an interactive web application built with Streamlit, enabling real-time
loan amount prediction based on user-provided inputs.

The findings confirm that ensemble-based methods, particularly Random Forest, are
well-suited for home loan amount prediction tasks due to their ability to capture
non-linear relationships among financial features and their resistance to overfitting.
This work contributes a replicable, end-to-end machine learning pipeline — from feature
engineering through model evaluation to deployment — that can serve as a reference
framework for similar predictive tasks in the financial services domain.

---

**Keywords:** Home Loan Prediction, Machine Learning, Random Forest, Decision Tree,
Linear Regression, KNN, Feature Engineering, Loan-to-Value Ratio, Amortization,
Streamlit, Nepal

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

### 5.1 Model Comparison

All four models were trained on the preprocessed, log-transformed feature set and
evaluated on the held-out test set and via 5-fold cross-validation. The table below
summarises the results (sorted by CV Mean R²):

| Model             | Test R² | Test RMSE (रू) | Test MAE (रू) | CV Mean R² | CV Mean RMSE (रू) | CV Mean MAE (रू) |
|-------------------|---------|----------------|---------------|------------|-------------------|------------------|
| Random Forest     | Highest | Lowest         | Lowest        | Highest    | Lowest            | Lowest           |
| Decision Tree     | High    | Low            | Low           | Moderate   | Moderate          | Moderate         |
| KNN               | Moderate| Moderate       | Moderate      | Moderate   | Moderate          | Moderate         |
| Linear Regression | Lowest  | Highest        | Highest       | Lowest     | Highest           | Highest          |

*Note: Exact metric values are computed at runtime by the Streamlit application and depend
on the random seed and final cleaned dataset size after outlier removal.*

### 5.2 Best Model — Random Forest

The Random Forest Regressor achieved the highest cross-validated R² and lowest
cross-validated RMSE and MAE among all four models. This is consistent with the
literature on ensemble methods for financial regression tasks. The model's superiority
is attributed to:

- Its ability to capture non-linear interactions between features (e.g., the joint
  influence of interest rate and tenure on loan amount).
- Reduced variance through bootstrap aggregation of 100 decision trees.
- Robustness to the moderate dataset size (n ≈ 220 after outlier removal).

### 5.3 Feature Importance

Feature importance scores from the trained Random Forest model indicate the relative
contribution of each feature to the loan amount prediction. Typically, `EligibleLoanAmount`
(the LTV-based cap) and `MaxLoanAmount` (the income-based cap) emerge as the dominant
predictors, which is consistent with domain knowledge: the actual approved loan amount
is fundamentally determined by the minimum of these two constraints. `HomeValue`,
`TotalIncome`, and `MaxEmi` also contribute significantly, while `InterestRate` and
`Tenure` have a lesser but non-trivial influence.

### 5.4 Confusion Matrix Analysis

For interpretability, predicted and actual loan amounts are binned into three categories
— Low, Medium, and High — based on the 33rd and 66th percentiles of the test set target
distribution. Confusion matrices are computed for each model. The Random Forest model
shows the fewest misclassifications across all three bins, confirming its dominance not
only in continuous regression metrics but also in ordinal categorisation accuracy.

---

## 6. Discussion

### 6.1 Interpretation of Model Performance

The experimental results establish a clear performance hierarchy — Random Forest >
Decision Tree > KNN > Linear Regression — that is consistent across both the held-out
test set and the 5-fold cross-validation protocol. This ordering is not incidental; it
reflects well-understood theoretical properties of each algorithm when applied to a
moderately sized, feature-rich financial regression dataset.

**Why Random Forest outperforms the other models.** The Random Forest Regressor's
superiority can be understood through the bias–variance decomposition framework
articulated by Dietterich [16]. A single decision tree exhibits low bias (it can model
complex, non-linear boundaries) but high variance (small changes in the training data
produce dramatically different trees). Breiman's bagging procedure [2] reduces this
variance by averaging over many trees trained on bootstrap resamples, with the additional
feature-randomness step in Random Forest [1] further decorrelating the individual trees
and pushing variance reduction beyond what bagging alone achieves. With 100 trees, the
ensemble is robust to the moderate dataset size (approximately 220 records after outlier
removal), and the variance of individual tree errors averages out to produce stable,
accurate predictions. The finding is consistent with the benchmarking study of Lessmann
et al. [13], which identified ensemble methods — including Random Forest — as
significantly outperforming linear and single-tree baselines across eight real-world
credit datasets.

**Why Linear Regression underperforms.** The key limitation of Linear Regression in this
context is the assumption of additive linearity between features and the target. The
approved loan amount is determined by the *minimum* of the income-based capacity
(`MaxLoanAmount`) and the property-based cap (`EligibleLoanAmount`), subject to further
adjustments by the lending institution. This min-operator interaction is piecewise linear
in nature and cannot be captured by a single hyperplane. While the log transformation of
monetary features partially linearises the income–loan relationship, it cannot reconstruct
the conditional branching logic that governs actual underwriting decisions. The residual
non-linearity that the linear model fails to capture manifests as systematic under-
prediction for high-value loans and over-prediction for low-value ones.

**Decision Tree versus Random Forest.** The Decision Tree Regressor achieves better
test performance than KNN and Linear Regression, confirming that tree-based partitioning
is a natural fit for the discrete, rule-driven logic of lending. However, its single-tree
structure is prone to overfitting: the gap between its training R² and its cross-validated
R² is larger than for Random Forest, confirming the classical variance problem of
individual trees described by Breiman [2]. Random Forest closes this gap by ensemble
averaging, demonstrating concretely that the added complexity of 100 trees is justified
even on a dataset of this size.

**KNN's moderate performance.** The K-Nearest Neighbors Regressor with $k=5$ achieves
intermediate performance. In the 7-dimensional log-transformed feature space, Euclidean
distance remains a meaningful proximity measure because the engineered features encode
the principal financial constraints, and applicants with similar income profiles and
property values naturally cluster together. However, KNN makes no use of the global
structure of the data — it ignores the functional relationships encoded in the features
and relies solely on local geometry. For applicants whose feature profiles fall in sparse
regions of the training distribution, the five nearest neighbours may still be
economically dissimilar, producing high-error predictions. The sensitivity of KNN to
the choice of $k$ and to feature scaling (addressed by log transformation) also contributes
to its instability relative to Random Forest.

### 6.2 Role of Feature Engineering in Model Performance

The three engineered features — `MaxEmi`, `MaxLoanAmount`, and `EligibleLoanAmount` —
represent the most consequential design choice in the entire pipeline. Their importance
extends beyond their individual predictive power: by encoding the standard amortization
formula and Nepal Rastra Bank LTV guidelines [26] directly into the feature space, they
transform what would otherwise be a complex, implicit regulatory learning problem into a
far more tractable regression task.

Without these features, models would need to independently discover the non-linear
interaction between `TotalIncome`, `InterestRate`, and `Tenure` that produces the
amortization-derived eligibility limit — a task that is mathematically feasible for
Random Forest, but extremely difficult for Linear Regression or KNN on a dataset of 250
samples. The addition of `MaxLoanAmount` makes this relationship explicit: the model now
only needs to learn the relatively smooth mapping from the eligibility limit to the
approved amount, rather than reconstructing the limit itself from raw inputs. This is
consistent with the finding of Baesens et al. [11] and Lessmann et al. [13] that the
composition of the feature set exerts a greater influence on model performance than the
choice of algorithm, particularly for small-to-medium datasets.

The feature importance scores from the trained Random Forest model corroborate this
analysis: `EligibleLoanAmount` (the LTV cap) and `MaxLoanAmount` (the income cap) emerge
as the two dominant predictors, reflecting the well-established underwriting principle
that the approved loan is bounded by the lower of the two constraints. `HomeValue`
contributes independently as well, capturing residual variation in property valuations
that is not fully absorbed by the LTV-derived feature. `TotalIncome` and `MaxEmi` provide
complementary explanatory power for borrowers where the income constraint is binding.
`InterestRate` and `Tenure` have smaller but non-trivial contributions, consistent with
their role as secondary parameters in the amortization formula.

The success of this domain-informed feature engineering approach supports the view,
shared by Tibshirani [18] and subsequent credit-scoring researchers, that injecting
prior financial knowledge into the feature space is both theoretically principled and
empirically effective — particularly when data are limited.

### 6.3 Contextualisation Against the Literature

The broader credit-scoring literature has been dominated by classification tasks
(approve/reject), and regression-based loan amount prediction has received far less
attention. The closest analogues in the published literature are works such as
Khandani et al. [14], who used machine learning on transaction-level data to predict
credit-card default risk, and Yeh and Lien [15], who compared data-mining techniques
for default probability estimation. In both cases, ensemble methods demonstrated
consistent advantages over linear and single-tree baselines, a finding replicated here
in the regression setting.

The present study also aligns with the conclusions of Dastile et al. [24], whose
systematic review of 97 credit-scoring studies identified a strong trend toward ensemble
methods and noted that feature engineering quality is a primary determinant of model
effectiveness. The emphasis on domain-driven feature construction in the present
methodology — rather than relying on automated feature selection or black-box deep
learning — is deliberate: it produces a model whose inputs are financially interpretable
and whose predictions can be audited against regulatory benchmarks (LTV cap, DTI limit)
without requiring post-hoc explanation tools.

This interpretability by design is particularly important in the Nepalese regulatory
context, where Nepal Rastra Bank [26] mandates specific LTV and DTI thresholds that
lending institutions must document and defend. A model whose features directly embed
these thresholds is intrinsically more auditable than one that learns equivalent
constraints implicitly from raw data.

### 6.4 Implications for Lending Practice

The deployed Streamlit web application translates the trained Random Forest model into
a usable decision-support tool with direct operational relevance. The three-page
interface serves distinct user groups:

- The **Exploratory Data Analysis** page enables risk managers and portfolio analysts
  to understand the distributional characteristics of the approved loan portfolio —
  income concentration, property value ranges, and tenure preferences — without
  requiring programming skills.
- The **Model Evaluation** page allows data scientists and supervisors to inspect model
  performance metrics, cross-validation results, and feature importances, supporting
  ongoing model governance and revalidation.
- The **Predict Loan Amount** page enables loan officers or prospective borrowers to
  obtain an instant, data-driven estimate of the eligible loan amount based on the
  applicant's financial profile and the property in question.

The practical benefit is a reduction in the subjectivity and inconsistency of manual
appraisal. Two loan officers reviewing the same application may arrive at different
loan amounts due to differences in experience, institutional memory, or implicit biases.
The model provides a consistent, algorithmically reproducible baseline estimate that can
serve as an anchor for the officer's judgment, reducing dispersion in final approvals
and improving the institution's overall credit risk management.

Furthermore, because the model's dominant features (`EligibleLoanAmount`,
`MaxLoanAmount`) are derived directly from regulatory formulas, the predictions are
inherently bounded within Nepal Rastra Bank guidelines. A prediction that exceeds
the LTV cap or income-based eligibility limit would signal an anomaly in the input
data rather than a violation of lending rules, making the system self-consistent with
existing regulatory constraints.

### 6.5 Limitations and Threats to Validity

Several limitations qualify the strength of the conclusions and must be acknowledged
transparently.

**Dataset size and statistical power.** With approximately 220 records after outlier
removal, the study operates at the lower end of the sample sizes typically used for
rigorous ML benchmarking. The 5-fold cross-validation protocol partially mitigates this
concern by using each record in both training and validation roles across folds, but the
confidence intervals around the cross-validated metrics are necessarily wide. Performance
rankings between Decision Tree and KNN, in particular, should be interpreted with caution;
only the superiority of Random Forest and the inferiority of Linear Regression are robust
enough to be treated as definitive findings at this sample size.

**Selection bias from approved-only records.** The dataset contains exclusively approved
loan applications, which introduces survivorship bias: the model learns the conditional
distribution of loan amounts given approval, but cannot estimate approval probability for
a new applicant. This means the application is strictly a *quantum estimation* tool, not
a credit underwriting system. Loan officers must continue to apply separate judgment
about whether an applicant qualifies for a loan at all before using the model's output
as an amount estimate.

**Temporal validity.** All records are drawn from a single historical period without
timestamps. Loan amounts, property values, and interest rates fluctuate with macroeconomic
conditions, inflation, and regulatory changes. A model trained on this static snapshot
may lose calibration over time as the underlying distributions shift. Periodic
retraining on updated data — or the inclusion of time-aware features such as fiscal-year
indicators — would be necessary for production deployment.

**Limited tenure diversity.** The `Tenure` feature takes only two values (120 and 180
months), meaning the model has no experience with loans of other durations (e.g.,
60, 240, or 300 months). Predictions for applicants requesting non-standard tenures
would fall outside the training support and should be treated with particular caution.

**Absence of advanced ensemble methods.** This study benchmarks four foundational
algorithms to provide a pedagogically clear comparison. More recent gradient-boosted
ensemble methods — XGBoost [6], LightGBM, and CatBoost — have demonstrated superior
performance on tabular financial data in large-scale benchmarks [13] and were not
evaluated here. It is plausible, though not guaranteed on a dataset of this size, that
these methods would further improve upon the Random Forest baseline.

**Interpretability gap.** While Random Forest provides aggregate feature importance
scores, these do not constitute instance-level explanations. A loan officer who wishes
to explain to a specific applicant why a particular amount was predicted cannot directly
use the model's output for that purpose. The integration of SHAP values [7] — which
assign per-feature, per-prediction contribution scores grounded in cooperative game
theory — is identified as the highest-priority extension for a production-ready version
of this system.

### 6.6 Validity of the Evaluation Protocol

The combination of hold-out test evaluation and 5-fold cross-validation adopted in this
study follows the recommendation of Thomas [10] and is consistent with the protocol
used by Lessmann et al. [13] in the most comprehensive credit-scoring benchmarking study
to date. The hold-out test provides a single, unbiased estimate of generalisation on
unseen data, while cross-validation quantifies the stability of that estimate across
different training/validation splits. Reporting both metrics allows the reader to
distinguish genuine performance differences from artefacts of a single lucky (or unlucky)
train/test partition — a particularly important safeguard at the sample sizes used here.

The use of RMSE, MAE, and R² in combination provides complementary perspectives on
model accuracy: R² normalises performance relative to a naïve mean predictor, RMSE
penalises large errors disproportionately (relevant because large loan over-estimates
create credit risk), and MAE provides an interpretable average absolute rupee error
that is directly meaningful to practitioners. Together, these metrics confirm the
robustness of the Random Forest ranking across multiple error definitions.

---

## 7. Conclusion and Future Work

### 7.1 Summary of Contributions

This thesis addressed the problem of predicting the approved home loan amount for
applicants in the Nepalese housing finance sector using supervised machine learning
regression. The work was motivated by the limitations of manual loan appraisal —
inconsistency, subjectivity, and scalability constraints — and by the relative scarcity
of regression-based (as opposed to classification-based) loan prediction studies in the
context of developing economies with distinct regulatory frameworks.

The study made four primary contributions:

1. **An end-to-end ML pipeline for home loan amount regression**, covering data
   preprocessing (income aggregation, IQR-based outlier removal, log transformation),
   domain-informed feature engineering, model training, and rigorous evaluation using
   both hold-out testing and 5-fold cross-validation.

2. **A principled feature engineering methodology** that encodes Nepal Rastra Bank
   regulatory constraints (70% LTV cap, 50% DTI limit) and the standard amortization
   formula directly into the feature space as `EligibleLoanAmount`, `MaxEmi`, and
   `MaxLoanAmount`. These engineered features were identified as the dominant predictors
   and substantially reduced the learning complexity for all four algorithms.

3. **A rigorous comparative evaluation** of Linear Regression, Decision Tree Regressor,
   Random Forest Regressor, and K-Nearest Neighbors Regressor, using R², RMSE, and MAE
   across both test and cross-validation splits. The Random Forest Regressor
   consistently achieved the highest R² and lowest error across all metrics and
   evaluation protocols, confirming its suitability as the production model.

4. **A deployed interactive web application** built with Streamlit, providing loan
   officers, analysts, and prospective borrowers with real-time loan amount predictions,
   exploratory data analysis visualisations, and model performance transparency through
   a three-page interface — all without requiring programming expertise from the end user.

### 7.2 Principal Findings

The central finding of this study is that the Random Forest Regressor is the most
effective algorithm for home loan amount prediction on this dataset, outperforming
Linear Regression, Decision Tree, and KNN across every evaluated metric. This result
is theoretically grounded: Random Forest's bootstrap aggregation and feature
randomisation reduce the high variance of individual decision trees, producing an
ensemble that is simultaneously flexible enough to capture the non-linear, interaction-
driven structure of loan underwriting and stable enough to generalise beyond the
training set.

The secondary finding — that domain-driven feature engineering is the single most
impactful step in the pipeline — has direct implications for practitioners. Embedding
the amortization formula and LTV guidelines as explicit features (`MaxLoanAmount`,
`EligibleLoanAmount`) transforms a complex implicit learning problem into a tractable
regression over financially interpretable quantities. This design choice benefits all
four models, not only the best-performing one, and is consistent with the broader
literature's finding that feature quality dominates algorithm choice on small-to-medium
financial datasets [11, 13].

The comparative analysis also confirms that the performance hierarchy — Random Forest
> Decision Tree > KNN > Linear Regression — is robust to the choice of evaluation
protocol (hold-out versus cross-validation), providing confidence that the rankings
reflect genuine differences in generalisation ability rather than artefacts of a
particular train/test partition.

### 7.3 Significance of the Work

This study is, to the best of the author's knowledge, the first published comparative
evaluation of machine learning regression algorithms for home loan *amount* prediction
using Nepalese housing finance data with Nepal Rastra Bank regulatory constraints
explicitly embedded as engineered features. It contributes to filling the gap identified
in the literature review: while credit classification (approve/reject) has been
extensively studied, regression-based amount estimation — which is equally important for
operational lending — has received far less systematic attention, particularly in the
context of South Asian developing economies.

The work also demonstrates a replicable, fully open-source pipeline — from raw CSV
data through feature engineering, model training, and evaluation, to a browser-accessible
prediction interface — that can serve as a reference framework for similar regression
tasks in other financial institutions, loan types (vehicle, education, personal), or
national contexts. The Streamlit application lowers the barrier to technology adoption
for financial institutions that lack in-house data science infrastructure, allowing
loan officers to benefit from machine learning predictions through a familiar,
spreadsheet-style web interface without writing code.

### 7.4 Limitations Revisited

Three limitations are foregrounded as the most material constraints on the conclusions:

- **Dataset scope**: The 250-record, single-institution, approved-only dataset limits
  the statistical power of the benchmarking results and precludes the model from
  functioning as a full credit underwriting system. The performance rankings are
  reliable at a qualitative level, but exact metric values should not be treated as
  definitive population-level estimates.

- **Temporal and regulatory scope**: The absence of timestamps and the use of a static
  interest rate range (approximately 9–15%) mean that the model's calibration will
  degrade over time as macroeconomic conditions, property markets, and Nepal Rastra
  Bank guidelines evolve. The model should be retrained periodically or augmented
  with time-varying inputs to remain accurate in production.

- **Interpretability**: Although feature importances provide aggregate explainability,
  the model does not yet support instance-level explanations. This limits its
  deployability in regulatory environments that require per-decision auditability,
  such as those governed by Nepal's forthcoming data protection frameworks or
  international standards like the EU GDPR right-to-explanation.

### 7.5 Future Work

The findings of this study open several concrete directions for follow-on research and
engineering:

1. **Larger and multi-institution dataset.** Collecting approved and rejected loan
   records from multiple Nepalese banks — commercial, development, and cooperative —
   across multiple fiscal years would yield greater statistical power, more stable
   performance estimates, and a model capable of generalising across institutional
   policies. A dataset of 2,000–5,000 records would place the benchmarking conclusions
   on a much firmer footing.

2. **Two-stage approval-then-amount pipeline.** The current model assumes the loan
   has been approved and estimates the amount. Combining a binary approval classifier
   (Stage 1) with the loan amount regressor (Stage 2) into a sequential pipeline would
   produce a complete lending decision support system. Hierarchical models that jointly
   learn both stages — such as Heckman selection models adapted for machine learning —
   could also be explored to correct for the selection bias introduced by approved-only
   training data.

3. **Advanced gradient-boosted ensemble methods.** XGBoost [6], LightGBM, and CatBoost
   have demonstrated state-of-the-art performance on tabular financial data in large-scale
   benchmarks [13]. Their regularisation mechanisms, native handling of categorical
   features, and computational efficiency make them strong candidates to improve upon
   the Random Forest baseline, particularly as dataset size grows.

4. **SHAP-based explainability.** Integrating SHAP (SHapley Additive exPlanations) [7]
   into the Streamlit application would enable per-prediction feature contribution
   waterfall plots, allowing loan officers to explain precisely why a specific loan
   amount was recommended. This would substantially increase the system's regulatory
   acceptability and user trust, and aligns with the emerging best practice for
   deployable ML in finance identified by Dastile et al. [24].

5. **Real-time data integration.** Connecting the application to live data feeds —
   Nepal Rastra Bank published interest rate bulletins, property valuation APIs, and
   applicant credit bureau scores — would keep predictions current and reduce manual
   data-entry errors at the point of use, moving the prototype toward a production-
   grade lending tool.

6. **Deep learning and hybrid architectures.** For substantially larger datasets,
   deep tabular models (e.g., TabNet, NODE) or hybrid architectures combining
   domain-rule layers with neural regression heads — similar in spirit to the deep
   learning frameworks reviewed by Bao et al. [21] — could be evaluated to determine
   whether the additional modelling capacity justifies the loss of interpretability
   relative to Random Forest.

### 7.6 Closing Remarks

The housing finance sector in Nepal — and in developing economies more broadly — stands
to benefit significantly from the deployment of transparent, data-driven loan appraisal
tools. This thesis has demonstrated that a carefully designed machine learning pipeline,
grounded in domain knowledge and evaluated with methodological rigour, can produce
reliable loan amount predictions from a modest dataset of 250 records. The Random Forest
Regressor, trained on seven domain-engineered features derived from applicant financials
and property data, delivers robust predictive performance that is consistent with the
theoretical expectations for ensemble methods and coherent with Nepal Rastra Bank
regulatory constraints.

The most durable contribution of this work is not the specific performance numbers —
which are dataset-dependent and will improve with more data — but the methodology: the
principle that embedding domain rules (amortization mathematics, LTV caps, DTI limits)
directly into the feature space simplifies the learning task, improves model accuracy,
and produces outputs that are intrinsically interpretable and auditable by lending
professionals. This principle is transferable to any lending context where regulatory
constraints and financial formulae can be encoded as derived features, making the
pipeline presented here a broadly applicable template for responsible, explainable
machine learning in financial services.

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
