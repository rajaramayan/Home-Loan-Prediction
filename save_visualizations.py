import os
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

img_dir = r"e:\Downloads\Abin thesis\images"
os.makedirs(img_dir, exist_ok=True)

df = pd.read_csv(r'e:\Downloads\Abin thesis\home_loan_data (1).csv')

# 1. Marital Status Distribution
plt.figure(figsize=(7, 5))
marital_counts = df['MartialStatus'].value_counts()
plt.bar(marital_counts.index, marital_counts.values, color='#4c72b0', edgecolor='black')
plt.title("Marital Status Distribution")
plt.xlabel("Marital Status")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(img_dir, "01_marital_status_distribution.png"), bbox_inches='tight', dpi=200)
plt.close()

# 2. Education Distribution
plt.figure(figsize=(7, 5))
education_counts = df['Education'].value_counts()
plt.bar(education_counts.index, education_counts.values, color='#55a868', edgecolor='black')
plt.title("Education Distribution")
plt.xlabel("Education")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(img_dir, "02_education_distribution.png"), bbox_inches='tight', dpi=200)
plt.close()

# 3. Age Group Distribution
df['AgeGroup'] = pd.cut(df['Age'], bins=[20, 30, 40, 50, 60], labels=['21-30', '31-40', '41-50', '51-60'])
age_counts = df['AgeGroup'].value_counts().sort_index()
plt.figure(figsize=(7, 5))
plt.bar(age_counts.index.astype(str), age_counts.values, color='#c44e52', edgecolor='black')
plt.title("Age Group Distribution")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(img_dir, "03_age_group_distribution.png"), bbox_inches='tight', dpi=200)
plt.close()

# 4. Home Value Distribution
df['HomeValueGroup'] = pd.cut(df['HomeValue'], bins=5)
home_counts = df['HomeValueGroup'].value_counts().sort_index()
plt.figure(figsize=(8, 5))
plt.bar(home_counts.index.astype(str), home_counts.values, color='#8172b1', edgecolor='black')
plt.title("Home Value Distribution")
plt.xlabel("Home Value Range")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(img_dir, "04_home_value_distribution.png"), bbox_inches='tight', dpi=200)
plt.close()

# 5. Loan Amount Distribution
df['LoanAmountGroup'] = pd.cut(df['LoanAmount'], bins=5)
loan_counts = df['LoanAmountGroup'].value_counts().sort_index()
plt.figure(figsize=(8, 5))
plt.bar(loan_counts.index.astype(str), loan_counts.values, color='#ccb974', edgecolor='black')
plt.title("Loan Amount Distribution")
plt.xlabel("Loan Amount Range")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(img_dir, "05_loan_amount_distribution.png"), bbox_inches='tight', dpi=200)
plt.close()

# 6. EMI Distribution
df['EmiGroup'] = pd.cut(df['Emi'], bins=5)
emi_counts = df['EmiGroup'].value_counts().sort_index()
plt.figure(figsize=(8, 5))
plt.bar(emi_counts.index.astype(str), emi_counts.values, color='#64b5cd', edgecolor='black')
plt.title("EMI Distribution")
plt.xlabel("EMI Range")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(img_dir, "06_emi_distribution.png"), bbox_inches='tight', dpi=200)
plt.close()

# Feature Engineering
df_fe = df.drop(['Gender', 'MartialStatus', 'Age', 'Education', 'LoanType', 'LoanStatus'], axis=1)
df_fe['TotalIncome'] = (df_fe['ClientIncome'] + df_fe['FamilyIncome']) / 12
df_fe = df_fe.drop(['ClientIncome', 'FamilyIncome'], axis=1)
df_fe['MaxEmi'] = df_fe['TotalIncome'] / 2

def calculate_max_loan(max_emi, interest_rate, tenure_months):
    if tenure_months <= 0 or interest_rate <= 0:
        return 0
    monthly_rate = interest_rate / 1200
    factor = (((1 + monthly_rate)**tenure_months - 1) / (monthly_rate * (1 + monthly_rate)**tenure_months))
    return max_emi * factor

df_fe['MaxLoanAmount'] = df_fe.apply(lambda row: calculate_max_loan(row['MaxEmi'], row['InterestRate'], row['Tenure']), axis=1)
df_fe['EligibleLoanAmount'] = 0.7 * df_fe['HomeValue']
df_fe['Tenure'] = df_fe['Tenure'].apply(lambda x: '0' if x == 120 else '1')

# 7-14. Scatter plots numeric features vs LoanAmount
num_features = ['HomeValue', 'Emi', 'InterestRate', 'Tenure', 'TotalIncome', 'MaxEmi', 'MaxLoanAmount', 'EligibleLoanAmount']

idx = 7
for feature in num_features:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df_fe, x=feature, y='LoanAmount')
    plt.title(f'LoanAmount vs {feature}')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, f"{idx:02d}_scatter_LoanAmount_vs_{feature}.png"), bbox_inches='tight', dpi=200)
    plt.close()
    idx += 1

# 15. Scatter plot with index labels
df_reset = df_fe.reset_index()
plt.figure(figsize=(10, 7))
scatter = plt.scatter(df_reset['HomeValue'], df_reset['LoanAmount'], c=df_reset.index, cmap='viridis', s=50)
for i, row in df_reset.iterrows():
    plt.text(row['HomeValue'], row['LoanAmount'], str(i), fontsize=8, alpha=0.7)
plt.title('LoanAmount vs HomeValue (Colored and Labeled by Index)')
plt.xlabel('Home Value')
plt.ylabel('Loan Amount')
plt.colorbar(scatter, label='Index')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "15_scatter_colored_labeled_index.png"), bbox_inches='tight', dpi=200)
plt.close()

# 16. HomeValue Boxplot with Outliers
plt.figure(figsize=(7, 5))
sns.boxplot(x=df_fe['HomeValue'])
plt.title('HomeValue with Outliers')
plt.xlabel('HomeValue')
plt.savefig(os.path.join(img_dir, "16_boxplot_HomeValue_with_outliers.png"), bbox_inches='tight', dpi=200)
plt.close()

# Remove Outliers of HomeValue
percentile25 = df_fe['HomeValue'].quantile(0.25)
percentile75 = df_fe['HomeValue'].quantile(0.75)
iqr = percentile75 - percentile25
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
df1 = df_fe[(df_fe['HomeValue'] >= lower_limit) & (df_fe['HomeValue'] <= upper_limit)]

# 17. HomeValue Boxplot After Removing Outliers
plt.figure(figsize=(7, 5))
sns.boxplot(x=df1['HomeValue'])
plt.title('HomeValue After Removing Outliers')
plt.savefig(os.path.join(img_dir, "17_boxplot_HomeValue_after_outliers.png"), bbox_inches='tight', dpi=200)
plt.close()

# 18. TotalIncome Boxplot with Outliers
plt.figure(figsize=(7, 5))
sns.boxplot(x=df1['TotalIncome'])
plt.title('TotalIncome with Outliers')
plt.savefig(os.path.join(img_dir, "18_boxplot_TotalIncome_with_outliers.png"), bbox_inches='tight', dpi=200)
plt.close()

# Remove Outliers of TotalIncome
percentile25 = df1['TotalIncome'].quantile(0.25)
percentile75 = df1['TotalIncome'].quantile(0.75)
iqr = percentile75 - percentile25
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
nf = df1[(df1['TotalIncome'] >= lower_limit) & (df1['TotalIncome'] <= upper_limit)]

# 19. TotalIncome Boxplot After Removing Outliers
plt.figure(figsize=(7, 5))
sns.boxplot(x=nf['TotalIncome'])
plt.title('TotalIncome After Removing Outliers')
plt.savefig(os.path.join(img_dir, "19_boxplot_TotalIncome_after_outliers.png"), bbox_inches='tight', dpi=200)
plt.close()

# 20. Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df_fe.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "20_correlation_matrix.png"), bbox_inches='tight', dpi=200)
plt.close()

# 21. Total Income Skewness Histogram
key_features = ['HomeValue', 'InterestRate', 'Tenure', 'TotalIncome', 'MaxEmi', 'MaxLoanAmount', 'EligibleLoanAmount']
X = nf[key_features]
plt.figure(figsize=(7, 5))
plt.hist(X['TotalIncome'], bins=30, color='#4c72b0', edgecolor='black')
plt.title("TotalIncome Distribution (Histogram)")
plt.xlabel("Total Income")
plt.ylabel("Frequency")
plt.savefig(os.path.join(img_dir, "21_total_income_histogram.png"), bbox_inches='tight', dpi=200)
plt.close()

print("Successfully saved 21 visualization PNG files to:", img_dir)
