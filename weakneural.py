# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 00:09:29 2019

@author: vedika barde
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pyod.utils.data import generate_data, get_outliers_inliers
 # instantiating 
bail = pd.read_csv('C:/Users/vedika barde/Desktop/BE_pro/gandu1.csv')
gender=1
coaccused=0
parity=1
period=0
difference=0
accused=1
proof=0
delay=0
X_new=[[gender,coaccused,parity,period,difference,accused,proof,delay]]
#Encoding
from sklearn.preprocessing import LabelEncoder
number=LabelEncoder()
bail['Gender']= number.fit_transform(bail['Gender'])
bail['Output']= number.fit_transform(bail['Output'])
#All features
X = bail.iloc[:, [3,5,10,12,13,16,17,18]].values
Y = bail.iloc[:, 20].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)

#feature scaling
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()
#adding the input layer and first hidden layer
classifier.add(Dense(output_dim=4,init='uniform',activation='relu',input_dim=8))
#add second hidden layer

classifier.add(Dense(output_dim=4,init='uniform',activation='relu'))
#add output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
y_pred=classifier.predict(X_test)
classifier.fit(X_train,Y_train,batch_size=15,nb_epoch=100)
y_pred=(y_pred>0.5)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)
