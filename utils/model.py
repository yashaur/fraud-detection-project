import streamlit as st
import joblib
import lightgbm
import time

@st.cache_resource
def load_model():
    start = time.time()
    model = joblib.load('model/lgbm.pkl')
    duration = time.time() - start
    print(f'Model took {duration:.2f}s to load')
    return model

@st.cache_data
def predict(_model, X, output = 'prob'):
    start = time.time()
    if output == 'prob':
        y_preds = _model.predict_proba(X)[:,1]
    elif output == 'pred':
        y_preds = _model.predict(X)
    duration = time.time() - start
    print(f'Model took {duration:.2f}s to predict')
    return y_preds

if __name__ == '__main__':

    import numpy as np
    from utils.data import load_preprocess, load_prediction_samples, preprocess_input
    import pandas as pd

    model = load_model()

    X = load_prediction_samples()

    X_df = pd.DataFrame()
    
    for row in X:
        row_preproc = preprocess_input(row)
        X_df = pd.concat([X_df, row_preproc])

    X_df.reset_index(inplace=True, drop = True)

    y_preds = np.round(pd.DataFrame(predict(model, X_df)) * 100, 2).rename(columns = {0: 'probability'})

    X_df_with_prob = pd.concat([X_df, y_preds], axis = 1).drop(columns = ['sin_hour', 'cos_hour'])

    X_df_with_prob.to_csv('data/sample_data_with_probs.csv', index = False, header = True)

    # flag = (y_preds[0] > .8)
    # # print(np.sum(y_preds < .5))
    # idx = list(y_preds[flag].index)

    # print(idx)

    # frauds = X.loc[idx].drop(columns = ['sin_hour', 'cos_hour'])

    # print(frauds)

    # for row in idx:
    #     print(frauds.loc[row].to_dict())
        

    