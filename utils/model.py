import joblib
import lightgbm
import time

def load_model():
    start = time.time()
    model = joblib.load('model/lgbm.pkl')
    duration = time.time() - start
    print(f'Model took {duration:.2f}s to load')
    return model

def predict(model, X):
    start = time.time()
    y_probs = model.predict_proba(X)[:,1]
    duration = time.time() - start
    print(f'Model took {duration:.2f}s to predict')
    return y_probs


if __name__ == '__main__':

    import numpy as np
    from utils.data import load_preprocess

    model = load_model()

    X, y = load_preprocess()

    y_preds = predict(model, X)

    print(np.where(y_preds[:10,1] > 0.5, 1, 0))