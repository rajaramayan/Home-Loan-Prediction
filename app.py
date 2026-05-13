import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Home Loan Prediction", page_icon="🏠", layout="wide")

DATA_PATH = "home_loan_data (1).csv"
MODEL_PATH = "random_regresser.joblib"

# ── Data Loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_raw_data():
    return pd.read_csv(DATA_PATH)

# ── Preprocessing & Training ───────────────────────────────────────────────────
@st.cache_resource
def preprocess_and_train():
    df_raw = load_raw_data()
    df = df_raw.copy()

    # Feature Engineering
    df = df.drop(['Gender', 'MartialStatus', 'Age', 'Education', 'LoanType', 'LoanStatus'], axis=1)
    df['TotalIncome'] = (df['ClientIncome'] + df['FamilyIncome']) / 12
    df = df.drop(['ClientIncome', 'FamilyIncome'], axis=1)
    df['MaxEmi'] = df['TotalIncome'] / 2

    def calc_max_loan(max_emi, interest_rate, tenure_months):
        if tenure_months <= 0 or interest_rate <= 0:
            return 0
        monthly_rate = interest_rate / 1200
        factor = ((1 + monthly_rate) ** tenure_months - 1) / (monthly_rate * (1 + monthly_rate) ** tenure_months)
        return max_emi * factor

    df['MaxLoanAmount'] = df.apply(
        lambda r: calc_max_loan(r['MaxEmi'], r['InterestRate'], r['Tenure']), axis=1)
    df['EligibleLoanAmount'] = 0.7 * df['HomeValue']

    # Encode Tenure: 120 months → 0, 180 months → 1
    df['Tenure'] = df['Tenure'].apply(lambda x: 0 if x == 120 else 1)

    # Remove HomeValue outliers
    p25, p75 = df['HomeValue'].quantile(0.25), df['HomeValue'].quantile(0.75)
    iqr = p75 - p25
    df1 = df[(df['HomeValue'] >= p25 - 1.5 * iqr) & (df['HomeValue'] <= p75 + 1.5 * iqr)]

    # Remove TotalIncome outliers
    p25, p75 = df1['TotalIncome'].quantile(0.25), df1['TotalIncome'].quantile(0.75)
    iqr = p75 - p25
    nf = df1[(df1['TotalIncome'] >= p25 - 1.5 * iqr) & (df1['TotalIncome'] <= p75 + 1.5 * iqr)]

    key_features = ['HomeValue', 'InterestRate', 'Tenure', 'TotalIncome',
                    'MaxEmi', 'MaxLoanAmount', 'EligibleLoanAmount']
    log_features = ['HomeValue', 'TotalIncome', 'MaxEmi', 'MaxLoanAmount', 'EligibleLoanAmount']

    y = nf['LoanAmount']
    X = nf[key_features]
    X_transform = X.copy()
    X_transform[log_features] = np.log1p(X[log_features])

    X_train, X_test, y_train, y_test = train_test_split(
        X_transform, y, test_size=0.2, random_state=42)

    model_dict = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42),
        'KNN': KNeighborsRegressor(n_neighbors=5),
    }

    results = {}
    k = 5
    for name, model in model_dict.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        cv_r2   = cross_val_score(model, X_transform, y, cv=k, scoring='r2')
        cv_mae  = cross_val_score(model, X_transform, y, cv=k, scoring='neg_mean_absolute_error')
        cv_rmse = cross_val_score(model, X_transform, y, cv=k, scoring='neg_root_mean_squared_error')
        results[name] = {
            'Test R²':      round(r2,   4),
            'Test RMSE':    round(rmse, 2),
            'Test MAE':     round(mae,  2),
            'CV Mean R²':   round(np.mean(cv_r2),    4),
            'CV Mean MAE':  round(-np.mean(cv_mae),  2),
            'CV Mean RMSE': round(-np.mean(cv_rmse), 2),
        }

    joblib.dump(model_dict['Random Forest'], MODEL_PATH)
    results_df = pd.DataFrame(results).T.sort_values('CV Mean R²', ascending=False)

    # Per-model test predictions (for comparison plots & confusion matrices)
    predictions = {}
    for name, model in model_dict.items():
        predictions[name] = model.predict(X_test)

    # Bin actual and predicted into Low / Medium / High for confusion matrices
    q33 = float(y_test.quantile(0.33))
    q66 = float(y_test.quantile(0.66))
    bins = [float(y_test.min()) - 1, q33, q66, float(y_test.max()) + 1]
    labels = ['Low', 'Medium', 'High']
    y_test_binned = pd.cut(y_test, bins=bins, labels=labels).astype(str)
    pred_binned = {
        name: pd.cut(
            pd.Series(preds, index=y_test.index).clip(bins[0] + 1, bins[-1] - 1),
            bins=bins, labels=labels
        ).astype(str)
        for name, preds in predictions.items()
    }

    return df_raw, nf, results_df, model_dict['Random Forest'], predictions, y_test, y_test_binned, pred_binned


# ── Helper: prediction pipeline ───────────────────────────────────────────────
def calc_max_loan_pred(max_emi, interest_rate, tenure_months):
    if tenure_months <= 0 or interest_rate <= 0:
        return 0
    monthly_rate = interest_rate / 1200
    factor = ((1 + monthly_rate) ** tenure_months - 1) / (monthly_rate * (1 + monthly_rate) ** tenure_months)
    return max_emi * factor

def prepare_input(home_value, total_income, interest_rate, tenure_months):
    log_features = ['HomeValue', 'TotalIncome', 'MaxEmi', 'MaxLoanAmount', 'EligibleLoanAmount']
    max_emi            = total_income / 2
    max_loan           = calc_max_loan_pred(max_emi, interest_rate, tenure_months)
    eligible_loan      = 0.7 * home_value
    tenure_encoded     = 0 if tenure_months == 120 else 1

    row = pd.DataFrame([{
        'HomeValue':          home_value,
        'InterestRate':       interest_rate,
        'Tenure':             tenure_encoded,
        'TotalIncome':        total_income,
        'MaxEmi':             max_emi,
        'MaxLoanAmount':      max_loan,
        'EligibleLoanAmount': eligible_loan,
    }])
    row[log_features] = np.log1p(row[log_features])
    return row


# ── Load / train ───────────────────────────────────────────────────────────────
with st.spinner("Loading data and training models…"):
    df_raw, nf, results_df, rf_model, predictions, y_test, y_test_binned, pred_binned = preprocess_and_train()

# ── Sidebar Navigation ─────────────────────────────────────────────────────────
st.sidebar.title("🏠 Home Loan Prediction")
page = st.sidebar.radio(
    "Navigate",
    ["📊 EDA", "📈 Model Evaluation", "🔮 Predict Loan Amount"]
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – EDA
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    st.markdown(f"Dataset: **{len(df_raw)} rows × {len(df_raw.columns)} columns**")

    with st.expander("📋 Raw Data Preview"):
        st.dataframe(df_raw.head(20), use_container_width=True)

    with st.expander("📐 Basic Statistics"):
        st.dataframe(df_raw.describe(), use_container_width=True)

    with st.expander("🔍 Missing Values"):
        missing = df_raw.isnull().sum().rename("Missing Count").to_frame()
        st.dataframe(missing, use_container_width=True)

    st.subheader("Categorical Distributions")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        counts = df_raw['MartialStatus'].value_counts()
        ax.bar(counts.index, counts.values, color='steelblue')
        ax.set_title("Marital Status Distribution")
        ax.set_xlabel("Marital Status")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots()
        counts = df_raw['Education'].value_counts()
        ax.bar(counts.index, counts.values, color='coral')
        ax.set_title("Education Distribution")
        ax.set_xlabel("Education")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close(fig)

    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots()
        df_tmp = df_raw.copy()
        df_tmp['AgeGroup'] = pd.cut(df_tmp['Age'], bins=[20, 30, 40, 50, 60],
                                    labels=['21-30', '31-40', '41-50', '51-60'])
        counts = df_tmp['AgeGroup'].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color='mediumseagreen')
        ax.set_title("Age Group Distribution")
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close(fig)

    with col4:
        fig, ax = plt.subplots()
        counts = df_raw['LoanStatus'].value_counts()
        ax.bar(counts.index, counts.values, color='mediumpurple')
        ax.set_title("Loan Status Distribution")
        ax.set_xlabel("Loan Status")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Numerical Distributions")
    col5, col6 = st.columns(2)

    with col5:
        fig, ax = plt.subplots()
        df_tmp2 = df_raw.copy()
        df_tmp2['LoanAmountGroup'] = pd.cut(df_tmp2['LoanAmount'], bins=5)
        counts = df_tmp2['LoanAmountGroup'].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color='tomato')
        ax.set_title("Loan Amount Distribution")
        ax.set_xlabel("Loan Amount Range")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close(fig)

    with col6:
        fig, ax = plt.subplots()
        df_tmp3 = df_raw.copy()
        df_tmp3['HomeValueGroup'] = pd.cut(df_tmp3['HomeValue'], bins=5)
        counts = df_tmp3['HomeValueGroup'].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color='dodgerblue')
        ax.set_title("Home Value Distribution")
        ax.set_xlabel("Home Value Range")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Correlation Heatmap")
    num_cols = df_raw.select_dtypes(include=np.number).columns
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_raw[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Box Plots – Outlier Detection")
    col7, col8 = st.columns(2)
    with col7:
        fig, ax = plt.subplots()
        sns.boxplot(y=df_raw['HomeValue'], ax=ax, color='lightblue')
        ax.set_title("HomeValue")
        st.pyplot(fig)
        plt.close(fig)
    with col8:
        fig, ax = plt.subplots()
        df_raw['TotalIncome_temp'] = (df_raw['ClientIncome'] + df_raw['FamilyIncome']) / 12
        sns.boxplot(y=df_raw['TotalIncome_temp'], ax=ax, color='lightsalmon')
        ax.set_title("Monthly Total Income")
        st.pyplot(fig)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Evaluation":
    st.title("📈 Model Evaluation")
    st.markdown("Models trained on **80%** of data after outlier removal and log-transformation.")

    st.subheader("Evaluation Metrics")
    st.dataframe(
        results_df.style.highlight_max(subset=['Test R²', 'CV Mean R²'], color='#c6efce')
                        .highlight_min(subset=['Test RMSE', 'Test MAE', 'CV Mean MAE', 'CV Mean RMSE'], color='#c6efce'),
        use_container_width=True
    )

    best = results_df.index[0]
    st.success(f"✅ Best Model: **{best}** (highest CV Mean R²: {results_df.loc[best, 'CV Mean R²']})")

    st.subheader("R² Score Comparison")
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#2ecc71' if m == best else '#3498db' for m in results_df.index]
    ax.barh(results_df.index, results_df['CV Mean R²'], color=colors)
    ax.set_xlabel("CV Mean R²")
    ax.set_title("Cross-Validated R² by Model")
    ax.set_xlim(0, 1)
    for i, v in enumerate(results_df['CV Mean R²']):
        ax.text(v + 0.005, i, f"{v:.4f}", va='center')
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("RMSE Comparison")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(results_df.index, results_df['CV Mean RMSE'], color='#e67e22')
    ax.set_xlabel("CV Mean RMSE (₹)")
    ax.set_title("Cross-Validated RMSE by Model")
    for i, v in enumerate(results_df['CV Mean RMSE']):
        ax.text(v + 500, i, f"₹{v:,.0f}", va='center')
    st.pyplot(fig)
    plt.close(fig)

    # ── Actual vs Predicted ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📌 Actual vs Predicted — All Models")
    model_names = list(predictions.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, name in enumerate(model_names):
        ax = axes[idx]
        y_pred = predictions[name]
        ax.scatter(y_test, y_pred, alpha=0.5, color='steelblue', s=25)
        mn = min(y_test.min(), y_pred.min())
        mx = max(y_test.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Perfect fit')
        ax.set_title(name)
        ax.set_xlabel("Actual Loan Amount (₹)")
        ax.set_ylabel("Predicted Loan Amount (₹)")
        r2 = results_df.loc[name, 'Test R²']
        ax.legend(title=f"R² = {r2}")
    plt.suptitle("Actual vs Predicted Loan Amount", fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Residual Plots ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📉 Residual Plots — All Models")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, name in enumerate(model_names):
        ax = axes[idx]
        residuals = np.array(y_test) - predictions[name]
        ax.scatter(predictions[name], residuals, alpha=0.5, color='darkorange', s=25)
        ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
        ax.set_title(name)
        ax.set_xlabel("Predicted Loan Amount (₹)")
        ax.set_ylabel("Residual (Actual − Predicted)")
    plt.suptitle("Residuals vs Predicted Values", fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Error Distribution ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Error Distribution — All Models")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    for idx, name in enumerate(model_names):
        ax = axes[idx]
        residuals = np.array(y_test) - predictions[name]
        ax.hist(residuals, bins=20, color='mediumpurple', edgecolor='white')
        ax.axvline(0, color='red', linestyle='--')
        ax.set_title(name)
        ax.set_xlabel("Residual (₹)")
        ax.set_ylabel("Frequency")
    plt.suptitle("Prediction Error Distribution", fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Confusion Matrices ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🟩 Confusion Matrices — Loan Amount Category (Low / Medium / High)")
    st.caption(
        "Since this is a regression problem, predicted loan amounts are binned into "
        "**Low** (bottom 33%), **Medium** (middle 33%), and **High** (top 33%) "
        "categories to visualise classification-like performance."
    )
    class_labels = ['Low', 'Medium', 'High']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, name in enumerate(model_names):
        ax = axes[idx]
        cm = confusion_matrix(y_test_binned, pred_binned[name], labels=class_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(name)
    plt.suptitle("Confusion Matrices — Loan Amount Category", fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Per-Class Accuracy Table ───────────────────────────────────────────────
    st.subheader("📋 Per-Class Accuracy Summary")
    acc_rows = []
    for name in model_names:
        cm = confusion_matrix(y_test_binned, pred_binned[name], labels=class_labels)
        for i, lbl in enumerate(class_labels):
            precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
            recall    = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
            acc_rows.append({'Model': name, 'Category': lbl,
                             'Precision': round(precision, 3), 'Recall': round(recall, 3)})
    acc_df = pd.DataFrame(acc_rows)
    st.dataframe(
        acc_df.style.background_gradient(subset=['Precision', 'Recall'], cmap='Greens'),
        use_container_width=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Loan Amount":
    st.title("🔮 Predict Home Loan Amount")
    st.markdown("Enter the applicant's details below to get an estimated loan amount using the **Random Forest** model.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            home_value = st.number_input(
                "🏡 Home Value (₹)", min_value=500_000, max_value=50_000_000,
                value=5_000_000, step=100_000,
                help="Market value of the property"
            )
            client_income = st.number_input(
                "💼 Client Monthly Income (₹)", min_value=5_000, max_value=1_000_000,
                value=50_000, step=1_000
            )
            family_income = st.number_input(
                "👨‍👩‍👧 Family Monthly Income (₹)", min_value=0, max_value=1_000_000,
                value=27_000, step=1_000
            )

        with col2:
            interest_rate = st.number_input(
                "📉 Annual Interest Rate (%)", min_value=5.0, max_value=20.0,
                value=10.3, step=0.1
            )
            tenure = st.selectbox(
                "📅 Loan Tenure",
                options=[120, 180],
                format_func=lambda x: f"{x} months ({x//12} years)"
            )

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted:
        total_income = client_income + family_income
        input_df = prepare_input(home_value, total_income, interest_rate, tenure)

        prediction = rf_model.predict(input_df)[0]

        st.markdown("---")
        st.subheader("📋 Prediction Results")

        r1, r2_col, r3 = st.columns(3)
        r1.metric("🏦 Predicted Loan Amount", f"₹{prediction:,.0f}")
        r2_col.metric("🏡 Home Value", f"₹{home_value:,.0f}")
        r3.metric("📊 LTV Ratio", f"{(prediction/home_value)*100:.1f}%")

        st.markdown("---")
        st.subheader("📐 Computed Intermediary Values")

        max_emi = total_income / 2
        eligible = 0.7 * home_value
        max_loan = calc_max_loan_pred(max_emi, interest_rate, tenure)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Monthly Total Income", f"₹{total_income:,.0f}")
        c2.metric("Max EMI Capacity", f"₹{max_emi:,.0f}")
        c3.metric("Max Loan (EMI-based)", f"₹{max_loan:,.0f}")
        c4.metric("Eligible Loan (70% LTV)", f"₹{eligible:,.0f}")

        st.info(
            f"The model predicts a loan amount of **₹{prediction:,.0f}** "
            f"for a property valued at **₹{home_value:,.0f}** "
            f"with a **{tenure}-month** tenure at **{interest_rate}%** annual interest."
        )
