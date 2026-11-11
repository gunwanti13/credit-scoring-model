import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score

TARGET = "SeriousDlqin2yrs"
L30 = "NumberOfTime30-59DaysPastDueNotWorse"
L60 = "NumberOfTime60-89DaysPastDueNotWorse"
L90 = "NumberOfTimes90DaysLate"

st.set_page_config(page_title="Credit Scoring Model", page_icon="💳")


def add_features(df):
    df = df.copy()
    df["TotalLatePayments"] = df[L30] + df[L60] + df[L90]
    df["HasLatePayment"] = (df["TotalLatePayments"] > 0).astype(int)
    df["SeverityScore"] = df[L30] + 2 * df[L60] + 3 * df[L90]
    df["MonthlyDebt"] = df["DebtRatio"] * df["MonthlyIncome"]
    df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
    df["IncomePerCreditLine"] = df["MonthlyIncome"] / (df["NumberOfOpenCreditLinesAndLoans"] + 1)
    df["LogIncome"] = np.log1p(df["MonthlyIncome"])
    return df


@st.cache_resource(show_spinner="Training the model (first run only)...")
def train_model():
    df = pd.read_csv("cs-training.csv", index_col=0)
    df = df[df["age"] > 18].copy()
    for c in ["MonthlyIncome", "NumberOfDependents"]:
        df[c] = df[c].fillna(df[c].median())
    df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(upper=1.5)
    df["DebtRatio"] = df["DebtRatio"].clip(upper=df["DebtRatio"].quantile(0.99))
    df["MonthlyIncome"] = df["MonthlyIncome"].clip(upper=df["MonthlyIncome"].quantile(0.99))
    df = add_features(df)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=20,
                                   class_weight="balanced", n_jobs=-1, random_state=42)
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {"ROC-AUC": roc_auc_score(yte, proba),
               "Precision": precision_score(yte, pred),
               "Recall": recall_score(yte, pred)}
    return model, list(X.columns), metrics


model, columns, metrics = train_model()

st.title("💳 Credit Scoring Model")
st.write("Enter an applicant's financial details to estimate the chance of "
         "serious delinquency in the next 2 years.")

c1, c2 = st.columns(2)
with c1:
    age = st.slider("Age", 21, 90, 40)
    income = st.number_input("Monthly income", 0, 500000, 5000, step=500)
    util = st.slider("Credit card utilization (0 to 1.5)", 0.0, 1.5, 0.3, 0.05)
    debt = st.slider("Debt ratio", 0.0, 2.0, 0.3, 0.05)
    deps = st.number_input("Dependents", 0, 15, 1)
with c2:
    late30 = st.number_input("Times 30-59 days late", 0, 50, 0)
    late60 = st.number_input("Times 60-89 days late", 0, 50, 0)
    late90 = st.number_input("Times 90+ days late", 0, 50, 0)
    lines = st.number_input("Open credit lines / loans", 0, 50, 6)
    realestate = st.number_input("Real estate loans", 0, 20, 1)

if st.button("Check Credit Risk", type="primary"):
    person = pd.DataFrame([{
        "RevolvingUtilizationOfUnsecuredLines": util, "age": age,
        L30: late30, "DebtRatio": debt, "MonthlyIncome": income,
        "NumberOfOpenCreditLinesAndLoans": lines, L90: late90,
        "NumberRealEstateLoansOrLines": realestate, L60: late60,
        "NumberOfDependents": deps}])
    person = add_features(person)[columns]
    p = model.predict_proba(person)[0, 1]
    st.subheader(f"Default probability: {p*100:.1f}%")
    st.progress(min(float(p), 1.0))
    if p >= 0.5:
        st.error("HIGH RISK (bad credit)")
    elif p >= 0.25:
        st.warning("MEDIUM RISK")
    else:
        st.success("LOW RISK (good credit)")

with st.expander("About the model"):
    st.write("Random Forest trained on the Give Me Some Credit dataset (150,000 borrowers).")
    st.write({k: round(v, 3) for k, v in metrics.items()})
