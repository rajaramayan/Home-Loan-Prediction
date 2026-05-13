# 🏠 Home Loan Amount Prediction

A machine learning project that predicts the eligible home loan amount for an applicant based on income, property value, interest rate, and tenure, built as part of a thesis entitled "Prediction of Home Loan Amount in RBBL using Machine Learning" in partial fulfillment of the requirements for the Degree of Master of Science in Information System Engineering at Purbanchal University School of Engineering, Nepal. The thesis is currently in progress and will be defended soon.


---

## 👤 Author
**Author:** Mrs. Avin Chettri  
**Thesis Supervisor:** Prof. Raj Kumar Thakur
**Email of Thesis Supervisor:** rajkshiva1@gmail.com .  Includes a full data analysis pipeline and an interactive **Streamlit** web application.

Repository: [rajaramayan/Home-Loan-Prediction](https://github.com/rajaramayan/Home-Loan-Prediction)

Date: 14 May 2026

---

## 📁 Project Structure

```
Home-Loan-Prediction/
│
├── home_loan_data (1).csv   # Raw dataset (250 records)
├── abin_thesis.py           # Full ML pipeline as a Python script
├── abin_thesis.ipynb        # Jupyter Notebook version of the pipeline
├── app.py                   # Streamlit web application
├── random_regresser.joblib  # Saved Random Forest model (generated on first run)
├── .gitignore
└── README.md
```

---

## 📊 Dataset

The dataset contains **250 records** of home loan applicants with the following features:

| Column | Description |
|---|---|
| `Gender` | Applicant gender |
| `MartialStatus` | Marital status |
| `Age` | Applicant age |
| `Education` | Education level |
| `ClientIncome` | Applicant's annual income |
| `FamilyIncome` | Family's annual income |
| `LoanType` | Type of loan |
| `HomeValue` | Market value of the property |
| `Emi` | Current EMI obligations |
| `InterestRate` | Annual interest rate (%) |
| `Tenure` | Loan tenure (120 or 180 months) |
| `LoanStatus` | Loan approval status |
| `LoanAmount` | **Target variable** — sanctioned loan amount |

---

## ⚙️ ML Pipeline

### 1. Exploratory Data Analysis
- Distribution plots for categorical features (Gender, Education, Marital Status, Loan Status)
- Age group, Home Value, Loan Amount, and EMI distribution charts
- Correlation heatmap
- Outlier detection using box plots

### 2. Feature Engineering
- Dropped non-predictive columns: `Gender`, `MartialStatus`, `Age`, `Education`, `LoanType`, `LoanStatus`
- Created `TotalIncome = (ClientIncome + FamilyIncome) / 12`
- Created `MaxEmi = TotalIncome / 2`
- Computed `MaxLoanAmount` using the standard loan eligibility formula:

$$MaxLoan = MaxEMI \times \frac{(1 + r)^n - 1}{r \cdot (1 + r)^n}$$

  where $r$ = monthly interest rate, $n$ = tenure in months

- Created `EligibleLoanAmount = 0.7 × HomeValue` (70% LTV)
- Encoded `Tenure`: 120 months → `0`, 180 months → `1`

### 3. Outlier Removal
- IQR-based outlier removal on `HomeValue` and `TotalIncome`

### 4. Normalisation
- Log transformation (`log1p`) applied to skewed features:  
  `HomeValue`, `TotalIncome`, `MaxEmi`, `MaxLoanAmount`, `EligibleLoanAmount`

### 5. Models Trained

| Model | Test R² | CV Mean R² |
|---|---|---|
| **Random Forest** ✅ | 0.9367 | Best |
| Linear Regression | 0.9339 | — |
| KNN | 0.9251 | — |
| Decision Tree | 0.8874 | — |

> **Best Model: Random Forest Regressor** (highest R² and lowest RMSE on both test set and 5-fold cross-validation)

---

## 🖥️ Streamlit App

The interactive web app has three pages:

### 📊 EDA
- Raw data preview and statistics
- Categorical and numerical distribution charts
- Correlation heatmap
- Box plots for outlier visualisation

### 📈 Model Evaluation
- Comparison table of all 4 models (Test & CV metrics)
- R² and RMSE bar charts with best model highlighted

### 🔮 Predict Loan Amount
- Input form: Home Value, Client Income, Family Income, Interest Rate, Tenure
- Outputs: Predicted Loan Amount, LTV Ratio, Max EMI, Eligible Loan Amount

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib streamlit
```

### Run the ML pipeline

```bash
python abin_thesis.py
```

### Launch the Streamlit app

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 📦 Requirements

| Package | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Visualisation |
| `scikit-learn` | ML models and evaluation |
| `scipy` | Skewness calculation |
| `joblib` | Model serialisation |
| `streamlit` | Web application |

---

## 📌 Sample Prediction

| Input | Value |
|---|---|
| Home Value | ₹50,00,000 |
| Monthly Total Income | ₹77,000 |
| Interest Rate | 10.3% |
| Tenure | 180 months (15 years) |
| **Predicted Loan Amount** | **₹29,78,201.90** |


