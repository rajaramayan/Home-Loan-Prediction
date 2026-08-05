# Methodology & Program Execution Visualizations

This document presents the complete methodology and step-by-step visual output produced during the execution of `abin_thesis.py` (and `abin_thesis.ipynb`). Each visualization is embedded and explained in the exact sequential order of program execution.

---

## 1. Exploratory Data Analysis (EDA) - Categorical & Demographic Distributions

During the initial phase of data exploration, categorical demographic variables are extracted from `home_loan_data (1).csv` and visualized using bar charts to understand the demographic profile of loan applicants.

### Output 1: Marital Status Distribution
![Marital Status Distribution](images/01_marital_status_distribution.png)
- **Code Segment**: `df['MartialStatus'].value_counts()`
- **Explanation**: This bar chart compares the count of married versus unmarried applicants in the dataset. Married applicants constitute the majority (~65%) of home loan applications, reflecting typical financial stability requirements for joint property ownership.

---

### Output 2: Education Distribution
![Education Distribution](images/02_education_distribution.png)
- **Code Segment**: `df['Education'].value_counts()`
- **Explanation**: Illustrates the educational background of borrowers categorized into four levels: *Plus two*, *Bachelor*, *SLC*, and *Master*. The majority of applicants possess *Plus two* or *Bachelor* level qualifications, indicating a broad mid-tier professional demographic.

---

### Output 3: Age Group Distribution
![Age Group Distribution](images/03_age_group_distribution.png)
- **Code Segment**: `df['AgeGroup'] = pd.cut(df['Age'], bins=[20, 30, 40, 50, 60], labels=['21-30', '31-40', '41-50', '51-60'])`
- **Explanation**: Applicants are binned into four age brackets. The peak applicant concentration lies in the `21-30` age range followed closely by `31-40`, representing early-to-mid career professionals seeking primary residential property loans.

---

## 2. Financial Feature Binned Distributions

To understand the continuous numeric variables, values are segmented into five equal-width intervals.

### Output 4: Home Value Distribution
![Home Value Distribution](images/04_home_value_distribution.png)
- **Code Segment**: `df['HomeValueGroup'] = pd.cut(df['HomeValue'], bins=5)`
- **Explanation**: Binned bar chart showing the frequency distribution of property collateral values. The bulk of properties are valued between 1.7M and 5.5M NPR/INR, with fewer high-end luxury properties in the upper tail.

---

### Output 5: Loan Amount Distribution
![Loan Amount Distribution](images/05_loan_amount_distribution.png)
- **Code Segment**: `df['LoanAmountGroup'] = pd.cut(df['LoanAmount'], bins=5)`
- **Explanation**: Frequency distribution of requested loan amounts. Clustered predominantly in the lower-to-middle ranges (1.0M to 3.5M), reflecting prudent borrowing aligned with applicant income profiles.

---

### Output 6: Monthly EMI Distribution
![EMI Distribution](images/06_emi_distribution.png)
- **Code Segment**: `df['EmiGroup'] = pd.cut(df['Emi'], bins=5)`
- **Explanation**: Distribution of Equated Monthly Installments (EMI) calculated for applicants, matching the underlying loan amount distribution.

---

## 3. Bivariate Relationship Analysis (Numeric Features vs. `LoanAmount`)

The script iterates through numeric features (`num_features`) using Seaborn scatter plots to assess linear dependencies, feature strength, and constraints relative to the target variable `LoanAmount`.

### Output 7: LoanAmount vs. HomeValue
![LoanAmount vs HomeValue](images/07_scatter_LoanAmount_vs_HomeValue.png)
- **Explanation**: Strong positive linear correlation showing that higher property values enable higher approved loan amounts. The upper limit strictly respects the 70% Loan-to-Value (LTV) rule (`EligibleLoanAmount = 0.7 * HomeValue`).

---

### Output 8: LoanAmount vs. EMI
![LoanAmount vs Emi](images/08_scatter_LoanAmount_vs_Emi.png)
- **Explanation**: Near-perfect linear alignment confirming that monthly installment commitments scale directly with the total principal loan amount borrowed.

---

### Output 9: LoanAmount vs. InterestRate
![LoanAmount vs InterestRate](images/09_scatter_LoanAmount_vs_InterestRate.png)
- **Explanation**: Scatter plot displaying loan amounts across discrete interest rate tiers offered by financial institutions (ranging from 9.51% to 10.37%).

---

### Output 10: LoanAmount vs. Tenure
![LoanAmount vs Tenure](images/10_scatter_LoanAmount_vs_Tenure.png)
- **Explanation**: Categorical scatter plot grouping loan amounts by loan duration in months (120 months / 10 years vs. 180 months / 15 years).

---

### Output 11: LoanAmount vs. TotalIncome
![LoanAmount vs TotalIncome](images/11_scatter_LoanAmount_vs_TotalIncome.png)
- **Explanation**: Plots requested loan amount against combined monthly household income (`(ClientIncome + FamilyIncome) / 12`), demonstrating positive income elasticity.

---

### Output 12: LoanAmount vs. MaxEmi
![LoanAmount vs MaxEmi](images/12_scatter_LoanAmount_vs_MaxEmi.png)
- **Explanation**: Compares requested loan amounts against the maximum permissible monthly EMI (`MaxEmi = TotalIncome / 2`), enforcing a 50% Debt-to-Income (DTI) cap.

---

### Output 13: LoanAmount vs. MaxLoanAmount
![LoanAmount vs MaxLoanAmount](images/13_scatter_LoanAmount_vs_MaxLoanAmount.png)
- **Explanation**: Compares actual requested loans against maximum theoretical borrowing capacity calculated via time-value-of-money annuity formulas.

---

### Output 14: LoanAmount vs. EligibleLoanAmount
![LoanAmount vs EligibleLoanAmount](images/14_scatter_LoanAmount_vs_EligibleLoanAmount.png)
- **Explanation**: Scatter plot verifying that requested loans remain under the eligible collateral threshold (`0.7 * HomeValue`).

---

### Output 15: Index-Annotated & Colored Scatter Plot
![LoanAmount vs HomeValue Index Colored](images/15_scatter_colored_labeled_index.png)
- **Code Segment**: `plt.scatter(df['HomeValue'], df['LoanAmount'], c=df.index, cmap='viridis', s=50)`
- **Explanation**: Enhanced scatter plot color-coded with the `viridis` palette and annotated with exact row index numbers. This allows precise visual identification of individual data points and potential bivariate outliers before cleaning.

---

## 4. Outlier Detection & Data Cleaning (IQR Method)

Outliers in continuous features are detected using the Interquartile Range (IQR) method and removed to prevent model distortion:
$$\text{Lower Limit} = Q1 - 1.5 \times \text{IQR}, \quad \text{Upper Limit} = Q3 + 1.5 \times \text{IQR}$$

### Output 16: HomeValue Box Plot (Before Outlier Removal)
![HomeValue with Outliers](images/16_boxplot_HomeValue_with_outliers.png)
- **Explanation**: Box plot showing raw `HomeValue` distribution. Points beyond the upper whisker represent extreme high-value property outliers.

---

### Output 17: HomeValue Box Plot (After Outlier Removal)
![HomeValue After Removing Outliers](images/17_boxplot_HomeValue_after_outliers.png)
- **Explanation**: Box plot after removing extreme records (`df1`). The feature distribution is bounded cleanly within normal variance limits.

---

### Output 18: TotalIncome Box Plot (Before Outlier Removal)
![TotalIncome with Outliers](images/18_boxplot_TotalIncome_with_outliers.png)
- **Explanation**: Box plot showing raw `TotalIncome` distribution with several high-income outlier points extending past the upper IQR threshold.

---

### Output 19: TotalIncome Box Plot (After Outlier Removal)
![TotalIncome After Removing Outliers](images/19_boxplot_TotalIncome_after_outliers.png)
- **Explanation**: Box plot of `TotalIncome` in the final cleaned dataset `nf`, confirming complete removal of extreme income outliers.

---

## 5. Correlation Matrix & Log Normalization

### Output 20: Feature Correlation Matrix Heatmap
![Correlation Matrix Heatmap](images/20_correlation_matrix.png)
- **Code Segment**: `sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")`
- **Explanation**: Annotated heatmap visualizing pairwise Pearson correlation coefficients ($r$). Highlights strong positive correlation between `HomeValue`, `EligibleLoanAmount`, `MaxLoanAmount`, and `LoanAmount`.

---

### Output 21: Total Income Distribution Histogram (Pre-Normalization)
![Total Income Histogram](images/21_total_income_histogram.png)
- **Code Segment**: `plt.hist(X['TotalIncome'], bins=30)`
- **Explanation**: 30-bin histogram revealing right-skewed income distribution. This justifies the log-transformation (`np.log1p`) applied in the script to normalize feature distributions before training regression models.

---

## 6. Algorithm Evaluation & Performance Comparison

Following feature pre-processing, four regression algorithms (*Linear Regression*, *Decision Tree*, *Random Forest*, and *KNN Regressor*) are trained and evaluated across holdout test sets and 5-fold cross-validation.

### Output 22: Machine Learning Algorithm Performance Comparison
![Algorithm Comparison](images/22_algorithm_comparison.png)
- **Code Segment**: `results_df = pd.DataFrame(results).T`
- **Explanation**: Multi-panel metric comparison chart evaluating all four models across four key dimensions:
  1. **Test $R^2$ Score**: Random Forest achieves the highest predictive power (**0.9367**), followed by Linear Regression (0.9339), KNN Regressor (0.9251), and Decision Tree (0.8874).
  2. **5-Fold Cross-Validation $R^2$**: Confirms generalization stability — Random Forest (**CV $R^2$ = 0.9413**), Linear Regression (0.9403), Decision Tree (0.9081), KNN Regressor (0.8726).
  3. **Test Mean Absolute Error (MAE)**: Random Forest achieves the lowest test MAE (**157,801 NPR**), followed by Linear Regression (161,640), KNN Regressor (165,946), and Decision Tree (198,195).
  4. **Test Root Mean Squared Error (RMSE)**: Random Forest achieves the lowest test RMSE (**201,173 NPR**), followed by Linear Regression (205,676), KNN Regressor (218,847), and Decision Tree (268,362).

---

## 7. Summary of Execution Outputs

| Output # | File Name | Feature / Analysis | Purpose |
|---|---|---|---|
| 01 | `01_marital_status_distribution.png` | `MartialStatus` | Demographic distribution check |
| 02 | `02_education_distribution.png` | `Education` | Education background analysis |
| 03 | `03_age_group_distribution.png` | `AgeGroup` | Age demographic segmentation |
| 04 | `04_home_value_distribution.png` | `HomeValueGroup` | Property value range binning |
| 05 | `05_loan_amount_distribution.png` | `LoanAmountGroup` | Loan request binning |
| 06 | `06_emi_distribution.png` | `EmiGroup` | Monthly installment binning |
| 07-14 | `07_scatter_...` to `14_scatter_...` | Bivariate Scatter Plots | Feature relationship analysis |
| 15 | `15_scatter_colored_labeled_index.png` | Annotated Scatter Plot | Outlier point identification |
| 16-17 | `16_boxplot_...` & `17_boxplot_...` | `HomeValue` Outliers | Pre/Post IQR cleaning |
| 18-19 | `18_boxplot_...` & `19_boxplot_...` | `TotalIncome` Outliers | Pre/Post IQR cleaning |
| 20 | `20_correlation_matrix.png` | Feature Matrix | Pairwise correlation evaluation |
| 21 | `21_total_income_histogram.png` | `TotalIncome` Histogram | Skewness inspection pre-log transform |
| 22 | `22_algorithm_comparison.png` | Algorithm Metrics | Model performance comparative evaluation |

