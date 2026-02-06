import pandas as pd
import numpy as np
import os
import time

def load_preprocess(data_path):

    start = time.time()

    X = pd.read_csv(
                        data_path + 'X_sample.csv',
                        dtype = {
                            'type': 'category',
                            'amount': 'float',
                            'oldbalanceOrg': 'float',
                            'newbalanceOrig': 'float',
                            'oldbalanceDest': 'float',
                            'newbalanceDest': 'float',
                            'hour_of_day': 'int8'
                        }
                        )

    X = X.assign(
                        sin_hour = np.sin(X['hour_of_day'].astype('int') * 2 * np.pi / 24),
                        cos_hour = np.cos(X['hour_of_day'].astype('int') * 2 * np.pi / 24)
    )

    y = pd.read_csv(data_path + 'y_sample.csv')

    duration = time.time() - start

    print(f'Data took {duration:.2f}s to load')

    return X, y

if __name__ == '__main__':
    
    data_path = './data/'

    X, y = load_preprocess(data_path)

    print(np.array(y).reshape(-1))