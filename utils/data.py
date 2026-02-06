import pandas as pd
import numpy as np
import os
import time

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
                            cos_hour = lambda df: np.cos(df['hour_of_day'].astype('int') * 2 * np.pi / 24)
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

if __name__ == '__main__':
    
    X, y = load_preprocess()

    print(X.head())