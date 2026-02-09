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
    from utils.data import load_preprocess
    import pandas as pd

    model = load_model()

    X, y = load_preprocess()

    y_preds = pd.DataFrame(predict(model, X))

    max_prob_idx = np.argmin(y_preds)

    sample = X.loc[max_prob_idx]

    print(sample.to_dict())