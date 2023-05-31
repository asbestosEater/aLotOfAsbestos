import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

df = pd.read_csv('delaney_solubility_with_descriptors.csv')

# print(df)

y = df['logS']
x = df.drop('logS', axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=100)

lr = LinearRegression()
lr.fit(X_train, Y_train)

y_lr_test_pred = lr.predict(X_test)

print(Y_test)
print(y_lr_test_pred)
print('Coefficient of determination (R^2): %.8f'
      % r2_score(Y_test, y_lr_test_pred))

