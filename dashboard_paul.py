import pandas as pd
import lightgbm
import joblib
import os
import numpy as np
import streamlit as st
import time
import pickle
import plotly.express as px


st.set_page_config(layout="wide")


path_data = os.path.join(os.getcwd(),'data')
path_model = os.path.join(os.getcwd(),'model')


st.title("Fraud Detection Model")

@st.cache_data
def load_data_x():
    return pd.read_csv(f'{path_data}/X_sample.csv',
                    dtype = {
                        'type': 'category',
                        'amount': 'float',
                        'oldbalanceOrg': 'float',
                        'newbalanceOrig': 'float',
                        'oldbalanceDest': 'float',
                        'newbalanceDest': 'float',
                        'hour_of_day': 'int8'
                    })

def load_data_y():
    return pd.read_csv(f'{path_data}/y_sample.csv')

@st.cache_resource
def load_model():
    with open(f"{path_model}/lgbm.pkl", "rb") as f:
        model = pickle.load(f)
    return model


# Loading the data
X = load_data_x()
y = load_data_y()

df = X.join(y)

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

# Display a checkbox with the label 'Show/Hide'

st.header("📊 Original Dataset Overview")

st.write("Preview of the uploaded dataset:")
st.dataframe(df)
# Naming the fraud column
TARGET_COL = "isFraud"

col1, col2 = st.columns(2)    

with col1:
    st.metric("Total Records", len(df))

with col2:
    st.metric("Fraud Cases", df["isFraud"].sum())

st.divider()

# Looking at basic stats

stat_data = (
    df.groupby(['hour_of_day'])['isFraud']
    .mean()
    .mul(100)
    .reset_index()
   
)

fig = px.bar(
    stat_data,
    x='hour_of_day',
    y='isFraud',
    title='Fraud Rate by Hour of Day',
    labels={
        'hour_of_day': 'Hour of the day',
        'isFraud':'Fraud rate'
    },
    color='isFraud'
)

fig.update_layout(
    title='Fraud Rate by Hour of Day',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    coloraxis_showscale=False,
)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)

st.markdown(
    '# The Statistics to look for!'
)

peak_hour = stat_data.loc[
    stat_data['isFraud'].idxmax(), 'hour_of_day'
]

st.markdown(
    f'### 🎭 Peak fraud risk observed at `{peak_hour}:00 hours`'
)

col1, col2 = st.columns(2)    

with col1:
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)

with col2:
    with st.container(border=True):
        top_hours = stat_data.sort_values('isFraud', ascending=False).reset_index(drop=True).head(10)
        # top_hours = top_hours.rename(columns={'hour_of_day':'Hour of the day',
        #                           'isFraud':'Fraud Rate (%)'})
        # top_hours['Fraud Rate (%)'] = (top_hours['Fraud Rate (%)']).round(2)
        
        # st.table(top_hours)

        st.markdown("##### Top Average Fraud Risk during different hours of the day")
        st.data_editor(
        top_hours,
        column_config={
            'hour_of_day':'Hour of the day',
            "isFraud": st.column_config.ProgressColumn(
                "Fraud Risk (%)",
                format="%.2f%%",
                min_value=0,
                max_value=100,
            ),
        },
        disabled=True
)

##############################################

col1, col2 = st.columns(2)

# Segment of the day 
df['time_segment'] = df['hour_of_day'].apply(
    lambda x: 'Night' if x >= 22 or x <= 5 else 'Day'
)
night_day = df.groupby('time_segment')['isFraud'].mean().mul(100).reset_index()
night_day['isFraud'] = night_day['isFraud']*100

fig2 =  px.bar(
    night_day, 
    x='time_segment', 
    y='isFraud',
    labels={
        'time_segment': 'Time Segment',
        'isFraud':'Fraud rate'
    },
    color='isFraud'
)

fig2.update_layout(
    title='Fraud Rate by the segment of the Day',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    coloraxis_showscale=False
)
# Payment type analysis
payment_fraud = (
    df.groupby('type')['isFraud']
    .mean()
    .reset_index()
    .rename(columns={
        'type': 'Payment Type',
        'isFraud': 'Fraud Rate'
    })
)

payment_fraud['Fraud Rate'] = payment_fraud['Fraud Rate'] * 100
payment_fraud = payment_fraud.sort_values('Fraud Rate', ascending=False)


fig3 = px.bar(
    payment_fraud,
    x='Payment Type',
    y='Fraud Rate',
    title='Fraud Rate by the type of transaction'
)

fig3.update_layout(
    title='Fraud Rate by the type of transaction',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    coloraxis_showscale=False
)

fig2.update_xaxes(showgrid=False)
fig2.update_yaxes(showgrid=False)
fig3.update_xaxes(showgrid=False)
fig3.update_yaxes(showgrid=False)


with col1:
    with st.container(border=True):
        st.plotly_chart(fig2)


with col2:
    with st.container(border=True):
        st.plotly_chart(fig3)




if st.checkbox("Show/Hide Top Frauds"):
    # ----------------------------
    # 4️⃣ Predict Probabilities
    # ----------------------------

    proba = model.predict_proba(data)

    # Add fraud probability column
    df["predicted_label"] = model.predict(data)
    df["fraud_probability"] = model.predict_proba(data)[:,1]
    df["prediction_status"] = df.apply(
        lambda x: "Correct" if x[TARGET_COL] == x["predicted_label"] else "Wrong",
        axis=1)


    # ----------------------------
    # 5️⃣ Show Top Fraud Alerts
    # ----------------------------
    st.header("🔥 Top Fraud Alerts")

    top_fraud = df.sort_values(
        by="fraud_probability",
        ascending=False
    ).head(10)

    st.dataframe(top_fraud)

