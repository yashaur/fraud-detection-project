import pandas as pd
import numpy as np
import os

data_path = './data/X_test.csv'

data = pd.read_csv(
                    data_path,
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

data = data.assign(
                    sin_hour = np.sin(data['hour_of_day'].astype('int') * 2 * np.pi / 24),
                    cos_hour = np.cos(data['hour_of_day'].astype('int') * 2 * np.pi / 24)
)