import pandas as pd
import streamlit as st
from utils.init import init_session_vars
from utils.data import (load_preprocess,
                        top_fraud_alerts,
                        transactions_per_hour,
                        transactions_per_segment,
                        transactions_per_type)
from utils.model import load_model
from utils.shap import load_explainer, shap_values
from utils.charts import (create_fraud_by_time_chart,
                          create_segment_type_chart,
                          create_shap_chart)
from utils.shap_init import shap_init

init_session_vars()

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("📊 Fraud Detection Dashboard")

X, y = st.session_state['X'].copy(), st.session_state['y'].copy()
transactions_per_hour_df = st.session_state['transactions_per_hour']
transactions_per_segment_df = st.session_state['transactions_per_segment']
transactions_per_type_df = st.session_state['transactions_per_type']
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
fraud_by_time = transactions_per_hour(top_fraud_alerts_df)
fraud_rate_by_time =  fraud_by_time / transactions_per_hour_df * 100

time_chart = create_fraud_by_time_chart(fraud_rate_by_time)

peak_hour = fraud_rate_by_time.sort_values(ascending = False).index[0]

st.markdown(f"### 🎭 Peak fraud risk observed during: `{peak_hour}`")

# # --------------------------------------------------
# # Segment Analysis
# # --------------------------------------------------

fraud_trans_by_segment = transactions_per_segment(top_fraud_alerts_df)

fraud_rate_by_segment = fraud_trans_by_segment / transactions_per_segment_df * 1000

fig_segment = create_segment_type_chart(fraud_rate_by_segment)

plot1, plot2 = st.columns(2)

with plot1:
    st.plotly_chart(time_chart, config = {"displayModeBar": False}, height = 'stretch')
with plot2:
    st.plotly_chart(fig_segment, config = {"displayModeBar": False}, height = 'stretch')


# # --------------------------------------------------
# # Payment Type Analysis
# # --------------------------------------------------


fraud_trans_by_type = transactions_per_type(top_fraud_alerts_df)
all_trans_by_type = pd.concat([fraud_trans_by_type, transactions_per_type_df], axis = 1).fillna(0).rename(columns={'type': 'Total Transactions', 'Transaction Type': 'Total Fraud Transactions'})
fraud_rate_by_type = round(all_trans_by_type["Total Fraud Transactions"] / all_trans_by_type["Total Transactions"] * 1000, 2)
fig_type = create_segment_type_chart(fraud_rate_by_type)

top_fraud_type = fraud_trans_by_type.index[fraud_trans_by_type == max(fraud_trans_by_type)].values[0]
st.markdown(f"### 💳 Most fraud detected for the transaction type: `{top_fraud_type}`")

c1, c2 = st.columns(2)

with c1:
    st.dataframe(all_trans_by_type, height = 350)

with c2:
    st.plotly_chart(fig_type, config = {"displayModeBar": False}, height = 350)


# # --------------------------------------------------
# # Global SHAP Importance
# # --------------------------------------------------
st.subheader("🧠 Global SHAP Importance")

with st.spinner('Loading SHAP values... Please wait for up to a minute without switching pages...'):
    shap_init()
    shap_values_df = st.session_state['shap_values']

    st.write('✅ SHAP Explainer loaded. Scroll below to see how each transaction feature contributed to the prediction model!')

    shap_chart = create_shap_chart(shap_values_df)

    st.plotly_chart(shap_chart, config = {"displayModeBar": False})