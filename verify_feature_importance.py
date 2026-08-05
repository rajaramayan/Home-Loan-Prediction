"""
verify_feature_importance.py
Verifies the ranked feature contributions cited in Section 5.5 of Thesis_report.md
against the saved Random Forest model and re-trained model on the actual dataset.
Runs fully headless (no plt.show() calls).
"""

import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Load and preprocess (mirrors abin_thesis.py exactly) ─────────────────────
df = pd.read_csv("home_loan_data (1).csv")

# Feature engineering
df["TotalIncome"] = (df["ClientIncome"] + df["FamilyIncome"]) / 12
df = df.drop(["Gender", "MartialStatus", "Age", "Education", "LoanType",
              "LoanStatus", "ClientIncome", "FamilyIncome"], axis=1)

df["MaxEmi"] = df["TotalIncome"] / 2


def calculate_max_loan(max_emi, interest_rate, tenure_months):
    if tenure_months <= 0 or interest_rate <= 0:
        return 0
    monthly_rate = interest_rate / 1200
    factor = (((1 + monthly_rate) ** tenure_months - 1)
              / (monthly_rate * (1 + monthly_rate) ** tenure_months))
    return max_emi * factor


df["MaxLoanAmount"] = df.apply(
    lambda row: calculate_max_loan(row["MaxEmi"], row["InterestRate"], row["Tenure"]),
    axis=1
)
df["EligibleLoanAmount"] = 0.7 * df["HomeValue"]
df["Tenure"] = df["Tenure"].apply(lambda x: "0" if x == 120 else "1")
df = df.reset_index()

# ── Outlier removal ───────────────────────────────────────────────────────────
for col in ["HomeValue", "TotalIncome"]:
    q25, q75 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q75 - q25
    df = df[(df[col] >= q25 - 1.5 * iqr) & (df[col] <= q75 + 1.5 * iqr)]

nf = df.copy()
print(f"Records after outlier removal: {len(nf)}")

# ── Feature / target split ────────────────────────────────────────────────────
key_features = ["HomeValue", "InterestRate", "Tenure", "TotalIncome",
                "MaxEmi", "MaxLoanAmount", "EligibleLoanAmount"]
log_features  = ["HomeValue", "TotalIncome", "MaxEmi", "MaxLoanAmount", "EligibleLoanAmount"]

X = nf[key_features].copy()
y = nf["LoanAmount"]

X_transform = X.copy()
X_transform[log_features] = np.log1p(X[log_features])

X_train, X_test, y_train, y_test = train_test_split(
    X_transform, y, test_size=0.2, random_state=42
)

# ── Train a fresh RF (same hyperparameters as abin_thesis.py) ─────────────────
rf_fresh = RandomForestRegressor(random_state=42)
rf_fresh.fit(X_train, y_train)

# ── Load saved model ──────────────────────────────────────────────────────────
rf_saved = joblib.load("random_regresser.joblib")

# ── Extract feature importances ───────────────────────────────────────────────
def report_importance(model_name, model, features):
    imp = model.feature_importances_
    pct = imp / imp.sum() * 100          # normalise to 100%
    ranked = sorted(zip(features, imp, pct), key=lambda x: x[2], reverse=True)
    print(f"\n{'='*60}")
    print(f" Feature Importance -- {model_name}")
    print(f"{'='*60}")
    print(f"{'Feature':<22} {'Raw Importance':>16} {'% Contribution':>16}")
    print("-" * 60)
    for feat, raw, pct_val in ranked:
        print(f"  {feat:<20} {raw:>16.6f} {pct_val:>15.1f}%")
    print("-" * 60)
    print(f"  {'TOTAL':<20} {imp.sum():>16.6f} {pct.sum():>15.1f}%")
    return ranked

ranked_fresh = report_importance("Freshly Trained RF (random_state=42)", rf_fresh, key_features)
ranked_saved = report_importance("Saved RF (random_regresser.joblib)",  rf_saved, key_features)

# ── Compare against thesis-reported figures ───────────────────────────────────
THESIS_CLAIMS = {
    "EligibleLoanAmount": 41.2,
    "MaxLoanAmount":       32.8,
    "HomeValue":           12.4,
}

print("\n" + "=" * 60)
print(" Verification Against Thesis Section 5.5 Claims")
print("=" * 60)

fresh_pct = {feat: pct for feat, _, pct in ranked_fresh}
saved_pct = {feat: pct for feat, _, pct in ranked_saved}

for feat, claimed in THESIS_CLAIMS.items():
    fresh_val = fresh_pct.get(feat, 0)
    saved_val = saved_pct.get(feat, 0)
    fresh_ok = "OK" if abs(fresh_val - claimed) <= 1.5 else "MISMATCH"
    saved_ok = "OK" if abs(saved_val - claimed) <= 1.5 else "MISMATCH"
    print(f"\n  {feat}")
    print(f"    Claimed in thesis : {claimed:.1f}%")
    print(f"    Fresh RF actual   : {fresh_val:.1f}%  [{fresh_ok}]")
    print(f"    Saved RF actual   : {saved_val:.1f}%  [{saved_ok}]")

# Group check: TotalIncome + MaxEmi
ti_me_fresh = fresh_pct.get("TotalIncome", 0) + fresh_pct.get("MaxEmi", 0)
ti_me_saved = saved_pct.get("TotalIncome", 0) + saved_pct.get("MaxEmi", 0)
print(f"\n  TotalIncome + MaxEmi (claimed 8.6%)")
print(f"    Fresh RF actual   : {ti_me_fresh:.1f}%  [{'OK' if abs(ti_me_fresh - 8.6) <= 1.5 else 'MISMATCH'}]")
print(f"    Saved RF actual   : {ti_me_saved:.1f}%  [{'OK' if abs(ti_me_saved - 8.6) <= 1.5 else 'MISMATCH'}]")

ir_t_fresh = fresh_pct.get("InterestRate", 0) + fresh_pct.get("Tenure", 0)
ir_t_saved = saved_pct.get("InterestRate", 0) + saved_pct.get("Tenure", 0)
print(f"\n  InterestRate + Tenure (claimed 5.0%)")
print(f"    Fresh RF actual   : {ir_t_fresh:.1f}%  [{'OK' if abs(ir_t_fresh - 5.0) <= 1.5 else 'MISMATCH'}]")
print(f"    Saved RF actual   : {ir_t_saved:.1f}%  [{'OK' if abs(ir_t_saved - 5.0) <= 1.5 else 'MISMATCH'}]")

print("\nDone.")
