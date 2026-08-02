import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

img_dir = r"e:\Downloads\Abin thesis\images"
os.makedirs(img_dir, exist_ok=True)

csv_path = r"e:\Downloads\Abin thesis\home_loan_data (1).csv"
df = pd.read_csv(csv_path)

df = df.drop(['Gender', 'MartialStatus', 'Age', 'Education', 'LoanType', 'LoanStatus'], axis=1)
df['TotalIncome'] = (df['ClientIncome'] + df['FamilyIncome']) / 12
df = df.drop(['ClientIncome', 'FamilyIncome'], axis=1)
df['MaxEmi'] = df['TotalIncome'] / 2

def calculate_max_loan(max_emi, interest_rate, tenure_months):
    if tenure_months <= 0 or interest_rate <= 0:
        return 0
    monthly_rate = interest_rate / 1200
    factor = (((1 + monthly_rate)**tenure_months - 1) / (monthly_rate * (1 + monthly_rate)**tenure_months))
    return max_emi * factor

df['MaxLoanAmount'] = df.apply(lambda row: calculate_max_loan(row['MaxEmi'], row['InterestRate'], row['Tenure']), axis=1)
df['EligibleLoanAmount'] = 0.7 * df['HomeValue']
df['Tenure'] = df['Tenure'].apply(lambda x: '0' if x == 120 else '1')

# Outliers removal HomeValue
percentile25 = df['HomeValue'].quantile(0.25)
percentile75 = df['HomeValue'].quantile(0.75)
iqr = percentile75 - percentile25
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
df1 = df[(df['HomeValue'] >= lower_limit) & (df['HomeValue'] <= upper_limit)]

# Outliers removal TotalIncome
percentile25 = df1['TotalIncome'].quantile(0.25)
percentile75 = df1['TotalIncome'].quantile(0.75)
iqr = percentile75 - percentile25
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
nf = df1[(df1['TotalIncome'] >= lower_limit) & (df1['TotalIncome'] <= upper_limit)]

y = nf['LoanAmount']
key_features = ['HomeValue', 'InterestRate', 'Tenure', 'TotalIncome', 'MaxEmi', 'MaxLoanAmount', 'EligibleLoanAmount']
X = nf[key_features]

log_features = ['HomeValue', 'TotalIncome', 'MaxEmi', 'MaxLoanAmount', 'EligibleLoanAmount']
X_transform = X.copy()
X_transform[log_features] = np.log1p(X[log_features])

X_train, X_test, y_train, y_test = train_test_split(X_transform, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'KNN Regressor': KNeighborsRegressor(n_neighbors=5)
}

results = {}
k = 5

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    cv_r2 = cross_val_score(model, X_transform, y, cv=k, scoring='r2')
    cv_mae = cross_val_score(model, X_transform, y, cv=k, scoring='neg_mean_absolute_error')
    cv_rmse = cross_val_score(model, X_transform, y, cv=k, scoring='neg_root_mean_squared_error')

    results[name] = {
        'Test R2': r2,
        'Test RMSE': rmse,
        'Test MAE': mae,
        'CV Mean R2': np.mean(cv_r2),
        'CV Mean MAE': -np.mean(cv_mae),
        'CV Mean RMSE': -np.mean(cv_rmse)
    }

results_df = pd.DataFrame(results).T
print("Algorithm Performance Metrics:")
print(results_df)

# Plotting Comparison Figures
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b1']

# Subplot 1: Test R2 Score
sns.barplot(x=results_df.index, y='Test R2', data=results_df, ax=axes[0, 0], palette=colors, edgecolor='black')
axes[0, 0].set_title('Model Comparison - Test R² Score (Higher is Better)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('R² Score')
axes[0, 0].set_ylim(0.7, 1.02)
for p in axes[0, 0].patches:
    axes[0, 0].annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10, fontweight='bold')

# Subplot 2: CV Mean R2 Score
sns.barplot(x=results_df.index, y='CV Mean R2', data=results_df, ax=axes[0, 1], palette=colors, edgecolor='black')
axes[0, 1].set_title('Model Comparison - 5-Fold CV Mean R² (Higher is Better)', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('CV R² Score')
axes[0, 1].set_ylim(0.7, 1.02)
for p in axes[0, 1].patches:
    axes[0, 1].annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10, fontweight='bold')

# Subplot 3: Test MAE
sns.barplot(x=results_df.index, y='Test MAE', data=results_df, ax=axes[1, 0], palette=colors, edgecolor='black')
axes[1, 0].set_title('Model Comparison - Test MAE (Lower is Better)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Mean Absolute Error')
for p in axes[1, 0].patches:
    axes[1, 0].annotate(f"{p.get_height():,.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10, fontweight='bold')

# Subplot 4: Test RMSE
sns.barplot(x=results_df.index, y='Test RMSE', data=results_df, ax=axes[1, 1], palette=colors, edgecolor='black')
axes[1, 1].set_title('Model Comparison - Test RMSE (Lower is Better)', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Root Mean Squared Error')
for p in axes[1, 1].patches:
    axes[1, 1].annotate(f"{p.get_height():,.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(img_dir, "22_algorithm_comparison.png"), bbox_inches='tight', dpi=200)
plt.close()

print("Successfully saved algorithm comparison PNG to:", os.path.join(img_dir, "22_algorithm_comparison.png"))
