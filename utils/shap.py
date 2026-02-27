import shap
import pandas as pd
import time

def load_explainer(model):
    return shap.TreeExplainer(model)

def shap_values(explainer, X):
    start = time.time()

    shap_values = list(zip(X.columns, explainer(X).values[0]))
    shap_values.sort(key = lambda val: val[1])
    shap_values = [(feature, shap) for feature, shap in shap_values if feature not in ['sin_hour', 'cos_hour']]
    shap_top, shap_bottom = shap_values[0][0], shap_values[-1][0]

    if __name__ == '__main__':
        for feature, shap in shap_values:
            print(f'{feature}: {shap:.2f}')

    duration = time.time() - start
    print(f'Explainer took {duration:.2f}s to produce SHAP values')

    return shap_top, shap_bottom


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
    shap_vals = shap_values(explainer, input)
    print(shap_vals)