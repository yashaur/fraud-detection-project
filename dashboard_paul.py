import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import shap

from utils.data import load_preprocess
from utils.model import load_model


# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("Fraud Detection Dashboard")


# --------------------------------------------------
# Load Data & Model (CACHED)
# --------------------------------------------------
@st.cache_data
def load_data():
    X, y = load_preprocess()
    return X, y


@st.cache_resource
def load_model_and_explainer():
    model = load_model()
    explainer = shap.TreeExplainer(model)
    return model, explainer


X, y = load_data()
model, explainer = load_model_and_explainer()

df = X.copy()
df["isFraud"] = y

hour_order = (
    pd.to_datetime(range(24), format="%H")
    .strftime("%I %p")
    .str.lstrip("0")
)

df["hour_12"] = pd.to_datetime(
    df["hour_of_day"], format="%H"
).dt.strftime("%I %p").str.lstrip("0")

df["hour_12"] = pd.Categorical(
    df["hour_12"],
    categories=hour_order,
    ordered=True
)

rename = {
        'type' : 'Type of Transaction',
        'amount' : 'Amount',
        'oldbalanceOrg' : 'Balance Before the Transaction : Origin',
        'newbalanceOrig' : 'Balance After the Transaction : Origin',
        'oldbalanceDest' : 'Balance Before the Transaction : Destination',
        'newbalanceDest' : 'Balance After the Transaction : Destination',
        'hour_12' : 'Hour of the Day',
        'hour_of_day' : 'Hour of Day',
        'fraud_probability' : 'Probability of Fraud'
    }


# --------------------------------------------------
# Top Fraud Alerts Section
# --------------------------------------------------
if st.checkbox("Show Top Fraud Alerts"):

    df_probs = df.copy()

    # Predict once
    df_probs["fraud_probability"] = model.predict_proba(X)[:, 1]

    st.header("🔥 Top Fraud Alerts")

    top_fraud = (
        df_probs
        .sort_values("fraud_probability", ascending=False)
        .head(10)
        .drop(columns=["isFraud", "sin_hour", "cos_hour", "hour_of_day"], errors="ignore")
        .rename(columns=rename)
    )

    st.dataframe(top_fraud, use_container_width=True)


# --------------------------------------------------
# Basic Metrics
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.metric("Total Records", len(df))

with col2:
    st.metric("Fraud Cases", int(df["isFraud"].sum()))

st.divider()


# --------------------------------------------------
# Fraud Rate by Hour
# --------------------------------------------------
stat_data = (
    df.groupby("hour_12")["isFraud"]
    .mean()
    .mul(100)
    .reset_index()
)

peak_hour = stat_data.loc[
    stat_data["isFraud"].idxmax(), "hour_12"
]

st.markdown(f"### 🎭 Peak fraud risk observed at `{peak_hour}`")

fig_hour = px.bar(
    stat_data,
    x="hour_12",
    y="isFraud",
    labels={"hour_of_day": "Hour", "isFraud": "Fraud Rate (%)"},
    color="isFraud"
)

fig_hour.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    coloraxis_showscale=False
)

st.plotly_chart(fig_hour, use_container_width=True)


# --------------------------------------------------
# Segment Analysis
# --------------------------------------------------
df["time_segment"] = df["hour_of_day"].apply(
    lambda x: "Night" if x >= 22 or x <= 5 else "Day"
)

segment_data = (
    df.groupby("time_segment")["isFraud"]
    .mean()
    .mul(100)
    .reset_index()
)

fig_segment = px.bar(
    segment_data,
    x="time_segment",
    y="isFraud",
    labels={"time_segment": "Segment", "isFraud": "Fraud Rate (%)"},
    color="isFraud"
)

st.plotly_chart(fig_segment, use_container_width=True)


# --------------------------------------------------
# Payment Type Analysis
# --------------------------------------------------
# Group and compute fraud rate
payment_fraud = (
    df.groupby("type", as_index=False)["isFraud"]
      .mean()
)

# Convert to percentage and round
payment_fraud["Fraud Rate (%)"] = (
    payment_fraud["isFraud"] * 100
).round(2)

# Drop original column
payment_fraud.drop(columns="isFraud", inplace=True)

# Rename transaction types
rename_type = {
    "TRANSFER": "Transfer",
    "CASH_OUT": "Cash Out",
    "CASH_IN": "Cash In",
    "DEBIT": "Debit",
    "PAYMENT": "Payment"
}

payment_fraud["type"] = (
    payment_fraud["type"]
        .astype(str)
        .map(rename_type)
        .fillna(payment_fraud["type"])
)

# Sort by fraud rate
payment_fraud.sort_values(
    "Fraud Rate (%)",
    ascending=False,
    inplace=True
)

# Plot
fig_payment = px.bar(
    payment_fraud,
    x="type",
    y="Fraud Rate (%)",
    color="Fraud Rate (%)",
    title="Fraud Rate by Payment Type"
)

fig_payment.update_layout(
    xaxis_title="Payment Type",
    yaxis_title="Fraud Rate (%)"
)

st.plotly_chart(fig_payment, use_container_width=True)



# --------------------------------------------------
# Global SHAP Importance
# --------------------------------------------------
st.subheader("🧠 Global SHAP Importance")


@st.cache_data
def compute_global_shap(_explainer, X_sample):
    shap_vals = _explainer.shap_values(X_sample)

    # Handle binary classification
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    importance = np.abs(shap_vals).mean(axis=0)

    shap_df = pd.DataFrame({
        "feature": X_sample.columns,
        "importance": importance
    }).sort_values("importance", ascending=False)

    return shap_df


# Compute SHAP
shap_df = compute_global_shap(explainer, X)

# Remove unwanted engineered features
shap_df = shap_df[~shap_df["feature"].isin(["cos_hour", "sin_hour"])]

# Rename the features as per standard English columns
shap_df["feature"] = shap_df["feature"].map(rename).fillna(shap_df["feature"])

# Plot
fig_shap = px.bar(
    shap_df.head(15),
    x="importance",
    y="feature",
    orientation="h",
    color="importance",
    title="Top Features Driving Fraud Risk"
)

st.plotly_chart(fig_shap, use_container_width=True)
