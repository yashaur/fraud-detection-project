import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import json

def convert_series_to_df(X):
    X_df = X.to_frame().T
    return X_df

@st.cache_data
def preprocess_input(X, source = 'raw'):

    start = time.time()

    if type(X) == dict:
        X = {k: [X[k]] for k in X}
        X_df = pd.DataFrame(X)
    
    elif type(X) == pd.DataFrame:
        X_df = X

    elif type(X) == pd.Series:
        X_df = convert_series_to_df(X)
    
    if source == 'app':
        X_df['hour_of_day'] -= 1

    X_df = X_df.assign(
                        sin_hour = np.sin(X_df['hour_of_day'].astype('int') * 2 * np.pi / 24),
                        cos_hour = np.cos(X_df['hour_of_day'].astype('int') * 2 * np.pi / 24)
                        )
    
    X_df['type'] = X_df['type'].str.upper()
    X_df['type'] = X_df['type'].str.replace(' ', '_')
    
    correct_order = [
                        'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                        'oldbalanceDest', 'newbalanceDest', 'hour_of_day',
                        'sin_hour', 'cos_hour'
                        ]
    
    
    X_df = X_df[correct_order].astype({
                            'type': 'category',
                            'amount': 'float',
                            'oldbalanceOrg': 'float',
                            'newbalanceOrig': 'float',
                            'oldbalanceDest': 'float',
                            'newbalanceDest': 'float',
                            'hour_of_day': 'int8',
                            'sin_hour': 'float',
                            'cos_hour': 'float'
    })

    categories = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
    X_df['type'] = pd.Categorical(X_df['type'], categories=categories)

    return X_df

@st.cache_data
def load_preprocess(which: str = 'both'):

    start = time.time()

    X = lambda: pd.read_csv(
                        'data/X_sample.csv',
                        dtype = {
                            'type': 'category',
                            'amount': 'float',
                            'oldbalanceOrg': 'float',
                            'newbalanceOrig': 'float',
                            'oldbalanceDest': 'float',
                            'newbalanceDest': 'float',
                            'hour_of_day': 'int8'
                        }
                        ).assign(
                            sin_hour = lambda df: np.sin(df['hour_of_day'].astype('int') * 2 * np.pi / 24),
                            cos_hour = lambda df: np.cos(df['hour_of_day'].astype('int') * 2 * np.pi / 24),
                        )

    y = lambda: pd.read_csv('data/y_sample.csv')

    def duration():
        dur = time.time() - start
        print(f'Data took {dur:.2f}s to load')

    if which == 'both':
        X_data, y_data = X(), y()
        duration()

        return X_data, y_data
    
    elif which == 'X':
        X_data = X()
        duration()

        return X_data
    
    elif which == 'y':
        y_data = y()
        duration()

        return y_data
    
@st.cache_data
def load_prediction_samples():
    start = time.time()
    with open('data/prediction_samples.json', 'r') as f:
        samples = json.load(f)
    dur = time.time() - start
    print(f'Prediction sample data took {dur:.2f}s to load')
    return samples

@st.cache_data
def top_fraud_alerts(X, threshold, _model):

    hours =  (
        pd.to_datetime(range(24), format="%H")
        .strftime("%I%p")
        .str.strip("0")
        )
    
    hours_range = hours + ' - ' + np.roll(hours, -1)

    hours_mapping = {hour: hour_range for hour, hour_range in zip(hours, hours_range)}

    X_df = X.copy()

    X_df["hour_12"] = (
                    pd.to_datetime(
                        X_df["hour_of_day"],
                        format="%H")
                        .dt.strftime("%I%p")
                        .str.lstrip("0")
                        .replace(hours_mapping)
                    )
    
    X_df['hour_12'] = pd.Categorical(
                                X_df['hour_12'],
                                categories = hours_range,
                                ordered = True
    )

    categories = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
    
    X_df['type'] = pd.Categorical(X_df['type'], categories=categories)


    y_probs = _model.predict_proba(X_df.drop(columns = 'hour_12'))[:,1]

    X_df['type'] = X_df['type'].str.replace('_', ' ').str.title()
    
    X_df['fraud_probability'] = y_probs * 100
    
    fraud_index = y_probs >= threshold

    X_df = (
            X_df[fraud_index]
            .drop(
                columns = ['sin_hour', 'cos_hour'])
            .sort_values(
                by = 'fraud_probability',
                ascending = False)
            .reset_index(drop=True)
        )
    
    X_df['fraud_probability'] = np.round(X_df['fraud_probability'], 2)

    X_df["time_segment"] = X_df["hour_of_day"].apply(
    lambda x: "Night" if x >= 22 or x <= 5 else "Day")
    
    return X_df

@st.cache_data
def transactions_per_hour(X):

    hours =  (
    pd.to_datetime(range(24), format="%H")
    .strftime("%I%p")
    .str.strip("0")
    )
    
    hours_range = hours + ' - ' + np.roll(hours, -1)

    hours_mapping = {hour: hour_range for hour, hour_range in zip(hours, hours_range)}

    X_df = X.copy()

    if 'hour_of_day' in X_df.columns:
        col = 'hour_of_day'
    else:
        col = 'Hour of Day'

    X_df["hour_12"] = (
                    pd.to_datetime(
                        X_df[col],
                        format="%H")
                        .dt.strftime("%I%p")
                        .str.lstrip("0")
                        .replace(hours_mapping)
                    )
    
    X_df['hour_12'] = pd.Categorical(
                                X_df['hour_12'],
                                categories = hours_range,
                                ordered = True
    )

    return X_df['hour_12'].value_counts().sort_index()

@st.cache_data
def transactions_per_segment(X):

    X_df = X.copy()

    if 'time_segment' not in X_df.columns:
        col = 'type'
        X_df["time_segment"] = X_df["hour_of_day"].apply(
        lambda x: "Night" if x >= 21 or x < 5 else "Day"
        )

    else:
        col = 'Time'

    trans_by_segment = (
                                    X_df
                                    .groupby("time_segment")[col]
                                    .count()
                        )
    
    return trans_by_segment

@st.cache_data
def transactions_per_type(X):

    X_df = X.copy()

    if 'type' in X_df.columns:
        col = 'type'
    elif 'Transaction Type' in X_df.columns:
        col = 'Transaction Type'

    trans_by_type = (
                                    X_df
                                    .groupby(col)[col]
                                    .count()
                        )
    try:
        trans_by_type.index = trans_by_type.index.str.replace('_', ' ').str.title()
    finally:
        return trans_by_type
    

if __name__ == '__main__':

    from utils.model import load_model
    
    X, y = load_preprocess()
    model = load_model()

    df = top_fraud_alerts(X,.5, model)

    print(df)

    # print(X.head())

    # test_input = {
    # 'type': 'Transfer', 'amount': 6961359.0, 'hour_of_day': 21, 
    # 'oldbalanceOrg': 6961359.0, 'newbalanceOrig': 0.0,
    # 'oldbalanceDest': 0.0, 'newbalanceDest': 0.0
    # }

    # output = preprocess_input(test_input)

    # print(output.dtypes)