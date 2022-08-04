# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 15:07:20 2019

@author: vedika barde
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
dataset = pd.read_csv('C:/Users/vedika barde/Desktop/BE_pro/bail_final.csv')
#handling the categorical data

from sklearn.preprocessing import LabelEncoder
number=LabelEncoder()
dataset['Gender']= number.fit_transform(dataset['Gender'].astype('str'))
dataset['Mental codition of accused']= number.fit_transform(dataset['Mental codition of accused'].astype('str'))
dataset['Nature of Alligations']= number.fit_transform(dataset['Nature of Alligations'].astype('str'))
dataset['Output']= number.fit_transform(dataset['Output'].astype('str'))
X = dataset.iloc[:, [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]].values
Y = dataset.iloc[:, 18].values
# Spliting the data set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20)  
regressor.fit(X_train, Y_train)  
y_pred = regressor.predict(X_test)  
print (metrics.accuracy_score(Y_test, y_pred.round()))
"""
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(regressor, X, Y, cv=10, scoring='accuracy')
print(scores)
print("RF")
print(scores.mean())
"""