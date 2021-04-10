import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
import seaborn as sns

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston.head()
boston['MEDV'] = boston_dataset.target

# correlation_matrix = boston.corr().round(2)
# sns.heatmap(data=correlation_matrix, annot=True)

X = boston['RM']
Y = boston['MEDV']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
lin_model = LinearRegression()
lin_model.fit(np.array(X_train).reshape(-1, 1), Y_train)
# lin_model.fit(X_train, Y_train)

# model evaluation for testing set
y_test_predict = lin_model.predict(np.array(X_test).reshape(-1, 1))
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)
print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

plt.figure(figsize=(10, 5))
plt.title("Regression Diagram")
plt.xlabel('RM (Room of Dwelling)')
plt.ylabel('MEDV (House Price)')
plt.scatter(X, Y, marker='o')
plt.scatter(X_test, y_test_predict, marker='x', c='r')
plt.plot(X_test, y_test_predict, c='m')

# model evaluation for training set
y_train_predict = lin_model.predict(np.array(X_train).reshape(-1, 1))
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)
print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
plt.show()