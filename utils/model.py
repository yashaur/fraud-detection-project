import joblib
import lightgbm
import time
# from utils.data import load_preprocess

def load_model():
    start = time.time()
    model = joblib.load('model/lgbm.pkl')
    duration = time.time() - start
    print(f'Model took {duration:.2f}s to load')
    return model

def predict(model, X):
    start = time.time()
    y_preds = model.predict_proba(X)[:,1]
    duration = time.time() - start
    print(f'Model took {duration:.2f}s to predict')
    return y_preds


if __name__ == '__main__':

    import numpy as np

    model = load_model('model/lgbm.pkl')

    X, y = load_preprocess('data/')

    y_preds = predict(model, X)

    print(np.where(y_preds[:10,1] > 0.5, 1, 0))