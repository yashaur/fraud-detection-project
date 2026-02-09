import pandas as pd
import numpy as np
import os
import time

def preprocess_input(X: dict):

    start = time.time()

    X = {k: [X[k]] for k in X}

    X_df = pd.DataFrame(X).assign(
                            sin_hour = lambda df: np.sin(df['hour_of_day'].astype('int') * 2 * np.pi / 24),
                            cos_hour = lambda df: np.cos(df['hour_of_day'].astype('int') * 2 * np.pi / 24)
                        )
    
    X_df['type'] = X_df['type'].str.upper()
    
    correct_order = [
                    'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                    'oldbalanceDest', 'newbalanceDest', 'hour_of_day',
                    'sin_hour', 'cos_hour'
                    ]
    
    X_df = X_df[correct_order].astype({
                            'type': 'category',
                            'amount': 'float',
                            'oldbalanceOrg': 'float',
                            'newbalanceOrig': 'float',
                            'oldbalanceDest': 'float',
                            'newbalanceDest': 'float',
                            'hour_of_day': 'int8',
                            'sin_hour': 'float',
                            'cos_hour': 'float'
    })

    return X_df


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
    
    # X, y = load_preprocess()

    # print(X.head())

    test_input = {
    'type': 'Transfer', 'amount': 6961359.0, 'hour_of_day': 21, 
    'oldbalanceOrg': 6961359.0, 'newbalanceOrig': 0.0,
    'oldbalanceDest': 0.0, 'newbalanceDest': 0.0
    }

    output = preprocess_input(test_input)

    print(output.dtypes)