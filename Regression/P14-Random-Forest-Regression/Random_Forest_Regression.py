#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:21:18 2019

@author: ubuntu
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataframe = "Random_Forest_Regression/Position_Salaries.csv"
df = pd.read_csv(dataframe)
X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y)

y_pred = regressor.predict([[6.5]])
print(y_pred)

X_grid = np.arange(min(X),max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()