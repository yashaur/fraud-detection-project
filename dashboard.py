import pandas as pd
import lightgbm
import joblib
import os
import numpy as np

model = joblib.load('./model/lgbm.pkl')

data = pd.read_csv('./data/X_test.csv',
                    dtype = {
                        'type': 'category',
                        'amount': 'float',
                        'oldbalanceOrg': 'float',
                        'newbalanceOrig': 'float',
                        'oldbalanceDest': 'float',
                        'newbalanceDest': 'float',
                        'hour_of_day': 'int8'
                    })

print(data.columns)

data = data.assign(
                    sin_hour = np.sin(data['hour_of_day'].astype('int') * 2 * np.pi / 24),
                    cos_hour = np.cos(data['hour_of_day'].astype('int') * 2 * np.pi / 24)
)

y_preds = model.predict_proba(data.head())

print(y_preds)