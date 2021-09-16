# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 10:06:13 2020

@author: Hassan Amer
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3]) 
onehotencoder = OneHotEncoder() 
categX = X[:, 3]
categX = X[:, 3].reshape(-1,1)
onehotencoder.fit(categX)
# onehotencoder = OneHotEncoder(categorical_features = [3])
categX = onehotencoder.transform(categX).toarray()
# Avoiding the Dummy Variable Trap
X = np.concatenate((X[:, :3],categX[:, 1:]), axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
# import statsmodels.formula.api as sm
import statsmodels.regression.linear_model as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]].astype(float)
X_opt = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

