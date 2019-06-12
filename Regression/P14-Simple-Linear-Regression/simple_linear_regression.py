#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:47:17 2019

@author: ubuntu
"""
#Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load Dataset
dataframe = "Simple_Linear_Regression/Salary_Data.csv"
df = pd.read_csv(dataframe)
X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

#Splitting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Fitting Simple Linear Regression
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting Test
y_pred = regressor.predict(X_test)

#Vizualising Training
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = "blue" )
plt.title("Salary vs Experience (Training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#Vizualising Test
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = "blue" )
plt.title("Salary vs Experience (Test set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()