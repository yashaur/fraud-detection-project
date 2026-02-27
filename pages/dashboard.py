import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from utils.init import init_session_vars
from utils.data import load_preprocess, top_fraud_alerts
from utils.model import load_model
from utils.shap import load_explainer

init_session_vars()

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("Fraud Detection Dashboard")

X, y = st.session_state['X'], st.session_state['y']
model = st.session_state['model']
explainer = st.session_state['explainer']
threshold = st.session_state['threshold'] / 100
field_names = st.session_state['field_names']
field_names['hour_12'] = 'Time'
field_names['fraud_probability'] = 'Probability of Fraud (%)'

top_fraud_alerts_df = top_fraud_alerts(X, threshold, model)
top_fraud_alerts_df = top_fraud_alerts_df.rename(columns = field_names)

# --------------------------------------------------
# Top Fraud Alerts Section
# --------------------------------------------------
st.header("🔥 Top Fraud Alerts")
st.warning(f"The threshold is currently set at {threshold*100:.2f}%")
st.dataframe(top_fraud_alerts_df, hide_index = True)


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
fraud_by_time = top_fraud_alerts_df['Time'].value_counts()

# st.markdown(f"### 🎭 Peak fraud risk observed at `{peak_hour}`")

# fig_hour = px.bar(
#     stat_data,
#     x="hour_12",
#     y="isFraud",
#     labels={"hour_of_day": "Hour", "isFraud": "Fraud Rate (%)"},
#     color="isFraud"
# )

# fig_hour.update_layout(
#     plot_bgcolor="rgba(0,0,0,0)",
#     paper_bgcolor="rgba(0,0,0,0)",
#     coloraxis_showscale=False
# )

# # --------------------------------------------------
# # Segment Analysis
# # --------------------------------------------------
# df["time_segment"] = df["hour_of_day"].apply(
#     lambda x: "Night" if x >= 22 or x <= 5 else "Day"
# )

# segment_data = (
#     df.groupby("time_segment")["isFraud"]
#     .mean()
#     .mul(100)
#     .reset_index()
# )

# fig_segment = px.bar(
#     segment_data,
#     x="time_segment",
#     y="isFraud",
#     labels={"time_segment": "Segment", "isFraud": "Fraud Rate (%)"},
#     color="isFraud"
# )



# plot1, plot2 = st.columns(2)

# with plot1:
#     st.plotly_chart(fig_hour)
st.write(fraud_by_time.to_dict())
st.plotly_chart(fraud_by_time.to_dict())
# with plot2:
#     st.plotly_chart(fig_segment)


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
