#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:48:33 2019

@author: Joelson Sartori Junior
"""

#Import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Load dataset
dataframe = "Multiple_Linear_Regression/50_Startups.csv"
df = pd.read_csv(dataframe)
X = df.iloc[:, :-1].values
y = df.iloc[:, 4].values

#Preprocessing the Dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features= [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy variable Trap
X = X[:, 1:]

#Splitting the Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2, random_state = 0)

#Fitting Multiple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting
y_pred = regressor.predict(X_test)

#|Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis=1)
X_opt = X[:, [0,1,2,3,4,5]]