import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

# ── Load Data ──────────────────────────────────────────────────────────────────
# from google.colab import drive
# drive.mount('/content/drive')

df = pd.read_csv('home_loan_data (1).csv')
df.head()

df

df.count()

df['Gender'].value_counts()

df['Education'].value_counts()

df['LoanStatus'].value_counts()

df['MartialStatus'].value_counts()

df['Tenure'].value_counts()

df.shape

df.columns

df.info()

df

df.tail()

df.shape

df.describe()

df.dtypes

# Check Duplicate Values
duplicate_rows = df.duplicated()
duplicate_rows

# Check Missing Values
df.isnull().sum()

df.isnull().sum()

df['LoanStatus'].value_counts()

df.head()

# ── Visualisations ─────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

marital_counts = df['MartialStatus'].value_counts()

plt.figure()
plt.bar(marital_counts.index, marital_counts.values)
plt.title("Marital Status Distribution")
plt.xlabel("Marital Status")
plt.ylabel("Count")
plt.show()

education_counts = df['Education'].value_counts()

plt.figure()
plt.bar(education_counts.index, education_counts.values)
plt.title("Education Distribution")
plt.xlabel("Education")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Create age groups
df['AgeGroup'] = pd.cut(df['Age'], bins=[20, 30, 40, 50, 60],
                        labels=['21-30', '31-40', '41-50', '51-60'])

age_counts = df['AgeGroup'].value_counts().sort_index()

plt.figure()
plt.bar(age_counts.index.astype(str), age_counts.values)
plt.title("Age Group Distribution")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.show()

df['HomeValueGroup'] = pd.cut(df['HomeValue'], bins=5)

home_counts = df['HomeValueGroup'].value_counts().sort_index()

plt.figure()
plt.bar(home_counts.index.astype(str), home_counts.values)
plt.title("Home Value Distribution")
plt.xlabel("Home Value Range")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

df['LoanAmountGroup'] = pd.cut(df['LoanAmount'], bins=5)

loan_counts = df['LoanAmountGroup'].value_counts().sort_index()

plt.figure()
plt.bar(loan_counts.index.astype(str), loan_counts.values)
plt.title("Loan Amount Distribution")
plt.xlabel("Loan Amount Range")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

df['EmiGroup'] = pd.cut(df['Emi'], bins=5)

emi_counts = df['EmiGroup'].value_counts().sort_index()

plt.figure()
plt.bar(emi_counts.index.astype(str), emi_counts.values)
plt.title("EMI Distribution")
plt.xlabel("EMI Range")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# ── Feature Engineering ────────────────────────────────────────────────────────

# Removing Unnecessary Features
df = df.drop(['Gender', 'MartialStatus', 'Age', 'Education', 'LoanType', 'LoanStatus'], axis=1)
df.head()

df['TotalIncome'] = df['ClientIncome'] + df['FamilyIncome']
df['TotalIncome'] = df['TotalIncome'] / 12
df.head()

# Removing FamilyIncome and ClientIncome
df = df.drop(['ClientIncome', 'FamilyIncome'], axis=1)
df.head()

df.head()

df.head()

df.head()

df.describe()

# Calculate MaxEMI
df['MaxEmi'] = df['TotalIncome'] / 2
df.head()

def calculate_max_loan(max_emi, interest_rate, tenure_months):
    if tenure_months <= 0 or interest_rate <= 0:  # Handle invalid values
        return 0
    monthly_rate = interest_rate / 1200  # Convert annual interest rate to monthly
    factor = ((1 + monthly_rate) ** tenure_months - 1) / (monthly_rate * (1 + monthly_rate) ** tenure_months)
    return max_emi * factor

df['MaxLoanAmount'] = df.apply(
    lambda row: calculate_max_loan(row['MaxEmi'], row['InterestRate'], row['Tenure']),
    axis=1
)

df.head()

df['EligibleLoanAmount'] = 0.7 * df['HomeValue']
df.head()

df.shape

# Encoding Tenure into Binary Values: 120 → 0, 180 → 1
df['Tenure'] = df['Tenure'].apply(lambda x: '0' if x == 120 else '1')
df.head()

df.head()

df.columns

num_features = ['HomeValue', 'Emi', 'InterestRate', 'Tenure',
                'TotalIncome', 'MaxEmi', 'MaxLoanAmount', 'EligibleLoanAmount']

# ── Scatter Plots: numeric features vs LoanAmount ─────────────────────────────
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

for feature in num_features:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x=feature, y='LoanAmount')
    plt.title(f'LoanAmount vs {feature}')
    plt.tight_layout()
    plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = df.reset_index()

plt.figure(figsize=(10, 7))
scatter = plt.scatter(df['HomeValue'], df['LoanAmount'], c=df.index, cmap='viridis', s=50)

for i, row in df.iterrows():
    plt.text(row['HomeValue'], row['LoanAmount'], str(i), fontsize=8, alpha=0.7)

plt.title('LoanAmount vs HomeValue (Colored and Labeled by Index)')
plt.xlabel('Home Value')
plt.ylabel('Loan Amount')
plt.colorbar(scatter, label='Index')
plt.grid(True)
plt.tight_layout()
plt.show()

# ── Outlier Handling ───────────────────────────────────────────────────────────

# HomeValue Outliers
import seaborn as sns
sns.boxplot(df['HomeValue'])
plt.title('HomeValue with Outliers')
plt.xlabel('HomeValue')
plt.show()

# Removing Outliers of HomeValue
percentile25 = df['HomeValue'].quantile(0.25)
percentile75 = df['HomeValue'].quantile(0.75)
iqr = percentile75 - percentile25
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
df[df['HomeValue'] > upper_limit]
df[df['HomeValue'] < lower_limit]
df1 = df[(df['HomeValue'] >= lower_limit) & (df['HomeValue'] <= upper_limit)]
df1.shape

import seaborn as sns
sns.boxplot(df1['HomeValue'])
plt.title('HomeValue After Removing Outliers')
plt.show()

import seaborn as sns
sns.boxplot(df1['TotalIncome'])
plt.title('TotalIncome with Outliers')
plt.show()

# Handling outlier of TotalIncome
percentile25 = df1['TotalIncome'].quantile(0.25)
percentile75 = df1['TotalIncome'].quantile(0.75)
iqr = percentile75 - percentile25
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
df1[df1['TotalIncome'] > upper_limit]
df1[df1['TotalIncome'] < lower_limit]
nf = df1[(df1['TotalIncome'] >= lower_limit) & (df1['TotalIncome'] <= upper_limit)]
nf.shape

import seaborn as sns
sns.boxplot(nf['TotalIncome'])
plt.title('TotalIncome with Outliers')
plt.show()

# ── Separating Output and Input Features ──────────────────────────────────────

df.head()

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

y = nf['LoanAmount']
y.shape

key_features = ['HomeValue', 'InterestRate', 'Tenure', 'TotalIncome', 'MaxEmi', 'MaxLoanAmount', 'EligibleLoanAmount']
key_features

X = nf[key_features]
X.columns

X.head()

X

# ── Normalisation Step ─────────────────────────────────────────────────────────

log_features = ['HomeValue', 'TotalIncome', 'MaxEmi', 'MaxLoanAmount', 'EligibleLoanAmount']
log_features

X_transform = X.copy()
X_transform.shape
X_transform.columns

X_transform.shape

from scipy.stats import skew
skewness = skew(X['TotalIncome'])
skewness

from scipy.stats import skew
skewness = skew(np.log1p(X['TotalIncome']))
skewness

import matplotlib.pyplot as plt
plt.hist(X['TotalIncome'], bins=30)
plt.show()

# Log transformation to log_features
X_transform[log_features] = np.log1p(X[log_features])
X_transform

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_transform, y, test_size=0.2, random_state=42)

X_test.shape

y_test.shape

X_train.shape

y_train.shape

# ── Model Training ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'knn': KNeighborsRegressor(n_neighbors=5)
}

# Dictionary to store evaluation results
results = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2 Score': r2
    }

results_df = pd.DataFrame(results).T
print("Model Evaluation Results:")
print(results_df)

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'knn': KNeighborsRegressor(n_neighbors=5)
}

# Dictionary to store evaluation results
results = {}

# Number of folds
k = 5

for name, model in models.items():

    # Train-Test Evaluation
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Cross-Validation Scores
    cv_r2 = cross_val_score(model, X, y, cv=k, scoring='r2')
    cv_mae = cross_val_score(model, X, y, cv=k, scoring='neg_mean_absolute_error')
    cv_rmse = cross_val_score(model, X, y, cv=k, scoring='neg_root_mean_squared_error')

    results[name] = {
        'Test R2': r2,
        'Test RMSE': rmse,
        'Test MAE': mae,
        'CV Mean R2': np.mean(cv_r2),
        'CV Mean MAE': -np.mean(cv_mae),
        'CV Mean RMSE': -np.mean(cv_rmse)
    }

results_df = pd.DataFrame(results).T
results_df = results_df.sort_values(by='CV Mean R2', ascending=False)
results_df

# Determine the best model based on R2 Score (higher is better)
best_model_name = max(results.items(), key=lambda x: x[1]['Test R2'])[0]
print(f"\nBest Model based on Test R2: {best_model_name}")

best_model_rmse = min(results.items(), key=lambda x: x[1]['Test RMSE'])[0]
print(f"Best Model based on Test RMSE: {best_model_rmse}")

# Save the Random Forest model
import joblib
joblib.dump(models['Random Forest'], 'random_regresser.joblib')

# Load the saved model
random_model = joblib.load('random_regresser.joblib')

# ── Prediction on New Data ─────────────────────────────────────────────────────
af = pd.DataFrame({
    'HomeValue': [5000000],
    'TotalIncome': [77000],
    'InterestRate': [10.3],
    'Tenure': [180]
})
af

af['MaxEmi'] = af['TotalIncome'] / 2
af['EligibleLoanAmount'] = af['HomeValue'] * 0.7
af

def calculate_max_loan(max_emi, interest_rate, tenure_months):
    monthly_rate = interest_rate / 12
    factor = ((1 + monthly_rate) ** tenure_months - 1) / (monthly_rate * (1 + monthly_rate) ** tenure_months)
    return max_emi * factor

af['MaxLoanAmount'] = af.apply(
    lambda row: calculate_max_loan(row['MaxEmi'], row['InterestRate'], row['Tenure']),
    axis=1
)
af

bf = af

bf['HomeValue'] = np.log1p(bf['HomeValue'])
bf

bf['TotalIncome'] = np.log1p(bf['TotalIncome'])
bf['MaxEmi'] = np.log1p(bf['MaxEmi'])
bf['EligibleLoanAmount'] = np.log1p(bf['EligibleLoanAmount'])
bf['MaxLoanAmount'] = np.log1p(bf['MaxLoanAmount'])
bf

f_features = ['HomeValue', 'InterestRate', 'Tenure', 'TotalIncome', 'MaxEmi', 'MaxLoanAmount', 'EligibleLoanAmount']
f_features

cf = bf[f_features]
cf

rf_prediction = random_model.predict(cf)
print("\n Prediction Results for New Single Data Point")
print(f"Random Forest Prediction (LoanAmount): {rf_prediction[0]:,.2f}")
