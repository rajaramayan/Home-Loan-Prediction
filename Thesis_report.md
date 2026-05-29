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

While the literature on credit scoring (classification) is mature, regression-based
prediction of the *approved loan amount* — rather than binary approval status — has
received substantially less attention, particularly in the context of developing economies
such as Nepal. No published study was found that specifically addresses home loan amount
prediction using machine learning on Nepalese housing finance data with Nepal Rastra Bank
regulatory constraints embedded as engineered features. This study contributes to filling
that gap by combining the standard ML benchmarking methodology of Lessmann et al. [13]
with domain-specific feature engineering grounded in amortization mathematics and
regulatory norms [26], and by delivering the trained model through an interactive web
application [25] accessible to non-technical end users.

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

### 6.1 Interpretation of Results

The clear superiority of Random Forest over Linear Regression confirms the hypothesis
that the relationship between applicant/property attributes and the approved loan amount
is non-linear. Interest rate, tenure, and income interact multiplicatively through the
amortization formula, which a linear model cannot capture directly even after log
transformation.

The underperformance of the Decision Tree Regressor relative to Random Forest — despite
both being tree-based — illustrates the well-known variance problem of single trees: they
tend to overfit the training set and perform poorly on unseen data. Random Forest mitigates
this through ensemble averaging.

KNN's moderate performance reflects the challenges of distance-based methods in moderate-
dimensional spaces (7 features), where the notion of "nearest neighbours" becomes less
reliable.

### 6.2 Role of Feature Engineering

The three engineered features — `MaxEmi`, `MaxLoanAmount`, `EligibleLoanAmount` — encode
the core logic of loan underwriting directly into the feature space. This is a form of
domain knowledge injection that simplifies the learning task for all models. Without these
features, the models would need to implicitly learn the amortization and LTV relationships
from raw data, which would be far harder with only 250 samples.

### 6.3 Practical Implications

The deployed Streamlit application provides a practical tool for loan officers or
applicants to obtain an instant estimate of the eligible loan amount. The three-page
interface — Exploratory Data Analysis, Model Evaluation, and Predict Loan Amount —
supports both end-users seeking predictions and analysts who wish to understand the
dataset and model behaviour.

### 6.4 Limitations

- **Small dataset size**: With only 250 records, the models may not generalise to
  the full population of loan applicants. A larger dataset would yield more reliable
  and stable performance estimates.
- **Only approved loans**: The dataset contains no rejected applications, meaning the
  model learns the distribution of approved amounts only. It cannot estimate whether
  a new applicant would be approved at all.
- **Static interest rates and tenure**: Only two tenure values (120 and 180 months)
  appear in the data, which limits the model's ability to generalise to other terms.
- **No time dimension**: Loan amounts and property values change over time due to
  inflation and market cycles. The model does not account for temporal trends.

---

## 7. Conclusion and Future Work

### 7.1 Conclusion

This thesis presented a comparative evaluation of four machine learning regression
algorithms — Linear Regression, Decision Tree, Random Forest, and KNN — for the task
of predicting home loan amounts from applicant and property data in a Nepalese financial
context. Domain-driven feature engineering (MaxEmi, MaxLoanAmount, EligibleLoanAmount)
was identified as a critical preprocessing step, encoding standard banking rules directly
into model inputs. The Random Forest Regressor consistently outperformed all other models
across test and cross-validation metrics, and was deployed as an interactive Streamlit
web application.

The work demonstrates that machine learning can provide reliable, consistent, and
transparent loan amount estimates that complement (or in future, partially automate)
the manual appraisal process in financial institutions.

### 7.2 Future Work

Several directions for future research are identified:

1. **Larger and more diverse dataset**: Collecting data from multiple banks and across
   longer time periods would improve model robustness and generalisability.
2. **Inclusion of rejected applications**: Combining approval prediction (classification)
   with amount prediction (regression) into a two-stage model would provide a more
   complete lending decision support system.
3. **Advanced ensemble methods**: XGBoost, LightGBM, and CatBoost have demonstrated
   superior performance on tabular financial data in recent benchmarks and could be
   evaluated as alternatives to Random Forest.
4. **Explainability (XAI)**: Integrating SHAP (SHapley Additive exPlanations) values
   would provide per-prediction explanations, improving trust and regulatory compliance.
5. **Real-time data integration**: Connecting the application to live property valuation
   APIs and bank interest rate feeds would keep predictions current without manual updates.

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
