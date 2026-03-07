import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from utils.init import init_session_vars
from utils.data import load_preprocess, top_fraud_alerts
from utils.model import load_model
from utils.shap import load_explainer
from utils.charts import create_fraud_by_time_chart

init_session_vars()

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("Fraud Detection Dashboard")

X, y = st.session_state['X'].copy(), st.session_state['y'].copy()
transactions_per_hour = st.session_state['transactions_per_hour']
model = st.session_state['model']
explainer = st.session_state['explainer']
threshold = st.session_state['threshold'] / 100
field_names = st.session_state['field_names'].copy()
field_names['hour_12'] = 'Time'
field_names['fraud_probability'] = 'Probability of Fraud (%)'

top_fraud_alerts_df = top_fraud_alerts(X, threshold, model)
top_fraud_alerts_df = top_fraud_alerts_df.rename(columns = field_names)

# --------------------------------------------------
# Top Fraud Alerts Section
# --------------------------------------------------
st.header("🔥 Top Fraud Alerts")
st.warning(f"The threshold is currently set at {threshold*100:.2f}%")
st.dataframe(top_fraud_alerts_df.drop(columns = ['Hour of Day', 'time_segment']), hide_index = True)


# --------------------------------------------------
# Basic Metrics
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.metric("Total Records", X.shape[0])

with col2:
    st.metric("Fraud Cases", top_fraud_alerts_df.shape[0])

st.divider()


# # --------------------------------------------------
# # Fraud Rate by Hour
# # --------------------------------------------------
fraud_by_time = top_fraud_alerts_df['Time'].value_counts().sort_index()
fraud_rate_by_time =  fraud_by_time / transactions_per_hour * 100
time_chart = create_fraud_by_time_chart(fraud_rate_by_time)
peak_hour = fraud_rate_by_time.sort_values(ascending = False).index[0]

st.markdown(f"### 🎭 Peak fraud risk observed from `{peak_hour}`")
st.plotly_chart(time_chart, config={"displayModeBar": False})


# # --------------------------------------------------
# # Segment Analysis
# # --------------------------------------------------

X["time_segment"] = X["hour_of_day"].apply(
    lambda x: "Night" if x >= 22 or x <= 5 else "Day"
)

total_trans_by_segment = (
    X.groupby("time_segment")['type']
    .count()
)

fraud_trans_by_segment = (
    top_fraud_alerts_df.groupby("time_segment")['Time']
    .count()
)

fraud_rate_by_segment = fraud_trans_by_segment / total_trans_by_segment * 10000

st.write(fraud_rate_by_segment)

fig_segment = px.bar(
    x=fraud_rate_by_segment.index,
    y=fraud_rate_by_segment.values,
    # labels={"time_segment": "Segment", "isFraud": "Fraud Rate (%)"},
    # color="isFraud"
)



plot1, plot2 = st.columns(2)

with plot1:
    st.plotly_chart(fig_segment)
with plot2:
    st.plotly_chart(fig_segment)


# # --------------------------------------------------
# # Payment Type Analysis
# # --------------------------------------------------
# # Group and compute fraud rate
# payment_fraud = (
#     df.groupby("type", as_index=False)["isFraud"]
#       .mean()
# )

# # Convert to percentage and round
# payment_fraud["Fraud Rate (%)"] = (
#     payment_fraud["isFraud"] * 100
# ).round(2)

# # Drop original column
# payment_fraud.drop(columns="isFraud", inplace=True)

# # Rename transaction types
# rename_type = {
#     "TRANSFER": "Transfer",
#     "CASH_OUT": "Cash Out",
#     "CASH_IN": "Cash In",
#     "DEBIT": "Debit",
#     "PAYMENT": "Payment"
# }

# payment_fraud["type"] = (
#     payment_fraud["type"]
#         .astype(str)
#         .map(rename_type)
#         .fillna(payment_fraud["type"])
# )

# # Sort by fraud rate
# payment_fraud.sort_values(
#     "Fraud Rate (%)",
#     ascending=False,
#     inplace=True
# )

# # Plot
# fig_payment = px.bar(
#     payment_fraud,
#     x="type",
#     y="Fraud Rate (%)",
#     color="Fraud Rate (%)",
#     title="Fraud Rate by Payment Type"
# )

# fig_payment.update_layout(
#     xaxis_title="Payment Type",
#     yaxis_title="Fraud Rate (%)"
# )

# st.plotly_chart(fig_payment, use_container_width=True)



# # --------------------------------------------------
# # Global SHAP Importance
# # --------------------------------------------------
# st.subheader("🧠 Global SHAP Importance")


# @st.cache_data
# def compute_global_shap(_explainer, X_sample):
#     shap_vals = _explainer.shap_values(X_sample)

#     # Handle binary classification
#     if isinstance(shap_vals, list):
#         shap_vals = shap_vals[1]

#     importance = np.abs(shap_vals).mean(axis=0)

#     shap_df = pd.DataFrame({
#         "feature": X_sample.columns,
#         "importance": importance
#     }).sort_values("importance", ascending=False)

#     return shap_df


# # Compute SHAP
# shap_df = compute_global_shap(explainer, X)

# # Remove unwanted engineered features
# shap_df = shap_df[~shap_df["feature"].isin(["cos_hour", "sin_hour"])]

# # Rename the features as per standard English columns
# shap_df["feature"] = shap_df["feature"].map(rename).fillna(shap_df["feature"])

# # Plot
# fig_shap = px.bar(
#     shap_df.head(15),
#     x="importance",
#     y="feature",
#     orientation="h",
#     color="importance",
#     title="Top Features Driving Fraud Risk"
# )

# st.plotly_chart(fig_shap, use_container_width=True)