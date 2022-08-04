# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:24:55 2019

@author: vedika barde
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
dataset=pd.read_csv("knn.csv")
x=dataset.iloc[:,[0,1]].values
y=dataset.iloc[:,2].values
#normal knn
model=KNeighborsClassifier(n_neighbors=3,weights="uniform")
model.fit(x,y)
y_pred=model.predict([[6,2]])
print(y_pred)
#distance locally
model=KNeighborsClassifier(n_neighbors=3,weights="distance")
model.fit(x,y)
y_pred=model.predict([[6,2]])
print(y_pred)