#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:33:46 2019

@author: ubuntu
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
dataset = "Decision_Tree_Regression/Position_Salaries.csv"
df = pd.read_csv(dataset)
X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state= 0)
regressor.fit(X,y)

y_pred = regressor.predict([[6.5]])
print(y_pred)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth ro Bluff')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()