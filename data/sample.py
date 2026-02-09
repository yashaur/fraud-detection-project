import pandas as pd
from sklearn.model_selection import train_test_split

X_test, y_test = pd.read_csv('data/X_test.csv'), pd.read_csv('data/y_test.csv')

X_sample, _, y_sample, _ = train_test_split(X_test, y_test, train_size=10000)

# X_sample.to_csv('data/X_sample.csv', index = False)
# y_sample.to_csv('data/y_sample.csv', index = False)

print(list(X_sample.type.value_counts().index))