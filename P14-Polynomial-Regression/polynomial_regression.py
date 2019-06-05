#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 08:43:40 2019

@author: ubuntu
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataframe = "Polynomial_Regression/Position_Salaries.csv"
df = pd.read_csv(dataframe)
X = df.iloc[:,1:2].values
y = df.iloc[:,2].values

#Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#Visualising Linear Regression
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Visualising Polynomial  Regression
#X_grid = np.arange(min(X), max(X), 0.1)
#X_grid = X_grid.reshape((len(X_grid), 1))
#plt.scatter(X, y, color='red')
#plt.plot(X_grid, lin_reg_2.predict( poly_reg.fit_transform(X_grid)), color = 'blue')
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
