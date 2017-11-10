# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 23:29:55 2017

@author: Raghav
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X =  dataset.iloc[:,:-1].values
y =  dataset.iloc[:,4].values

#Encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lb = LabelEncoder()
X[:, 3] = lb.fit_transform(X[:, 3])
OHotEncoder = OneHotEncoder(categorical_features= [3])
X = OHotEncoder.fit_transform(X).toarray()

#TO avoid the dummy variable trap
X = X[:, 1:]

#Spliting the dataset into training and test data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=0)

#Fitting multiple linear regression to the training test
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) 

# Predicting the Test set results
y_pred = regressor.predict(X_test)



#Building an optimal model using backward elimination
import statsmodels.formula.api as sm
X = np.append(arr= np.ones((50,1)).astype(int) , values= X, axis = 1 )
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
 
