import streamlit as st
import shap
import pandas as pd
import numpy as np
import time

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

@st.cache_data
def shap_values(_explainer, X, top_bottom = True, make_df = False):
    start = time.time()

    single_input = (X.shape[0] == 1)
    
    if single_input:
        shap_values = list(zip(X.columns, _explainer(X).values[0]))
        
    else:
        shap_values = _explainer(X).values
        shap_values = np.mean(np.abs(shap_values), axis = 0)
        shap_values = list(zip(X.columns, shap_values))
    
    shap_values.sort(key = lambda val: val[1], reverse = True)
    shap_values = [(feature, shap) for feature, shap in shap_values if feature not in ['sin_hour', 'cos_hour']]

    if top_bottom:
        shap_top, shap_bottom = shap_values[0][0], shap_values[-1][0]


    if __name__ == '__main__':
        if single_input:
            for feature, shap in shap_values:
                print(f'{feature}: {shap:.2f}')

    duration = time.time() - start
    print(f'Explainer took {duration:.2f}s to produce SHAP values')

    if top_bottom:
        return shap_top, shap_bottom
    else:
        if make_df:
            shap_values_df = pd.DataFrame({col: [val] for col, val in shap_values})
            return shap_values_df
        else:
            return shap_values


if __name__ == '__main__':
    import pandas as pd
    from utils.data import load_preprocess, preprocess_input, convert_series_to_df
    from utils.model import load_model, predict

    X = load_preprocess(which = 'X')
    model = load_model()
    y_probs = predict(model, X)

    sample = {"type": "TRANSFER", "amount": 418871.88, "oldbalanceOrg": 418871.88, "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "hour_of_day": 5}

    input = preprocess_input(sample)

    # print(input)

    explainer = load_explainer(model)
    shap_vals = shap_values(explainer, X, top_bottom=False, make_df = True)
    print(shap_vals)
    print(shap_vals.shape)
    print(type(shap_vals))
    # print(X.loc[0])