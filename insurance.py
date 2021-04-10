import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
import seaborn as sns

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def process(param, y_title, plot):
  dataset = pd.read_csv('insurance.csv')

  X = dataset[param]
  Y = dataset['expenses']
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
  lin_model = LinearRegression()
  lin_model.fit(np.array(X_train).reshape(-1, 1), Y_train)

  y_test_predict = lin_model.predict(np.array(X_test).reshape(-1, 1))
  rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
  r2 = r2_score(Y_test, y_test_predict)
  print("***************************************")
  print("Diagram untuk Pengukuran ", y_title)
  print("***************************************")
  print("The model performance for testing set")
  print("--------------------------------------")
  print('RMSE is {}'.format(rmse))
  print('R2 score is {}'.format(r2))
  print("\n")

  plt.subplot(2, 1, plot)
  plt.xlabel(y_title)
  plt.ylabel('Biaya Asuransi')
  plt.scatter(X, Y, marker='o')
  plt.scatter(X_test, y_test_predict, marker='x', c='r')
  plt.plot(X_test, y_test_predict, c='m')

  y_train_predict = lin_model.predict(np.array(X_train).reshape(-1, 1))
  rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
  r2 = r2_score(Y_train, y_train_predict)
  print("The model performance for training set")
  print("--------------------------------------")
  print('RMSE is {}'.format(rmse))
  print('R2 score is {}'.format(r2))
  print()
  print()

plt.figure(figsize=(10, 7))
process('age', 'Usia', 1)
process('bmi', 'Index Massa Berat Badan', 2)
plt.show()
