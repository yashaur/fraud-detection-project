import joblib
import lightgbm
import time

def load_model():
    start = time.time()
    model = joblib.load('model/lgbm.pkl')
    duration = time.time() - start
    print(f'Model took {duration:.2f}s to load')
    return model

def predict(model, X, output = 'prob'):
    start = time.time()
    if output == 'prob':
        y_preds = model.predict_proba(X)[:,1]
    elif output == 'pred':
        y_preds = model.predict(X)
    duration = time.time() - start
    print(f'Model took {duration:.2f}s to predict')
    return y_preds


if __name__ == '__main__':

    import numpy as np
    from utils.data import load_preprocess, load_prediction_samples, preprocess_input
    import pandas as pd

    model = load_model()

    X = load_preprocess(which = 'X')

    y_preds = pd.DataFrame(predict(model, X))
    flag = (y_preds[0] > .8)
    # print(np.sum(y_preds < .5))
    idx = list(y_preds[flag].index)

    print(idx)

    frauds = X.loc[idx].drop(columns = ['sin_hour', 'cos_hour'])

    print(frauds)

    for row in idx:
        print(frauds.loc[row].to_dict())
        

    