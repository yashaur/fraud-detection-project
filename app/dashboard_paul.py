import pandas as pd
import lightgbm
import joblib
import os
import numpy as np
import streamlit as st
import time
import pickle
import plotly.express as px


path_data = os.path.join(os.getcwd(),'datasets')
path_model = os.path.join(os.getcwd(),'model')


st.title("Fraud Detection Model")

@st.cache_data
def load_data():
    return pd.read_csv(f'{path_data}/sample_data.csv',
                    dtype = {
                        'type': 'category',
                        'amount': 'float',
                        'oldbalanceOrg': 'float',
                        'newbalanceOrig': 'float',
                        'oldbalanceDest': 'float',
                        'newbalanceDest': 'float',
                        'hour_of_day': 'int8'
                    })

@st.cache_resource
def load_model():
    with open(f"{path_model}/lgbm.pkl", "rb") as f:
        model = pickle.load(f)
    return model


# Loading the data
df = load_data()

# Loading the model
model = load_model()

# Selecting only the features
data = df.drop(columns='isFraud')

print(data.columns)

data = data.assign(
                    sin_hour = np.sin(data['hour_of_day'].astype('int') * 2 * np.pi / 24),
                    cos_hour = np.cos(data['hour_of_day'].astype('int') * 2 * np.pi / 24)
)

y_preds = model.predict_proba(data.head())

print(y_preds)

st.header("📊 Original Dataset Overview")

st.write("Preview of the uploaded dataset:")
st.dataframe(df)

col1, col2 = st.columns(2)


with col1:
    st.metric("Total Records", len(df))

with col2:
    st.metric("Fraud Cases", df["isFraud"].sum())

st.divider()


# ----------------------------
# 4️⃣ Predict Probabilities
# ----------------------------

# ⚠️ IMPORTANT:
# X must be the SAME features used during training
# X = df.drop(columns=["is_fraud"])   # adjust to your case

proba = model.predict_proba(data)

# Add fraud probability column
data["fraud_probability"] = proba[:, 1]

# ----------------------------
# 5️⃣ Show Top Fraud Alerts
# ----------------------------
st.header("🔥 Top Fraud Alerts")

top_fraud = data.sort_values(
    by="fraud_probability",
    ascending=False
).head(10)

st.dataframe(top_fraud)

# Looking at basic stats

stat_data = (
    df.groupby(['hour_of_day'])['isFraud']
    .mean()
    .reset_index()
   
)

fig = px.bar(
    stat_data,
    x='hour_of_day',
    y='isFraud',
    title='Fraud Rate by Hour of Day'
)

fig.update_layout(
    title='Fraud Rate by Hour of Day',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    coloraxis_showscale=False
)

st.markdown(
    # The Statistics too look for!
)
st.plotly_chart(fig, use_container_width=True)