# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:07:29 2019

@author: vedika barde
"""

import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
bail = pd.read_csv('C:/Users/vedika barde/Desktop/BE_pro/Bailfinal.csv')
from sklearn.preprocessing import LabelEncoder
number=LabelEncoder()
bail['Gender']= number.fit_transform(bail['Gender'])
bail['Output']= number.fit_transform(bail['Output'])
X = bail.iloc[:, [3,5,12,13,16,18]].values
Y = bail.iloc[:, 20].values
from scipy import stats
z = np.abs(stats.zscore(X))
threshold = 3
print(np.where(z > 3))
fbail_1= bail[(z < 3).all(axis=1)]
print(bail.shape)
print(fbail_1.shape)
#All features
X = fbail_1.iloc[:, [3,5,12,13,16,18]].values
Y = fbail_1.iloc[:, 20].values
x = StandardScaler().fit_transform(X)
from sklearn.model_selection import train_test_split
X_train3, X_test3, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)

#feature scaling
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train3)
X_test=sc_x.transform(X_test3)

#feature scaling
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()
#adding the input layer and first hidden layer
classifier.add(Dense(output_dim=4,init='uniform',activation='relu',input_dim=6))
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
from sklearn.externals import joblib
joblib.dump(classifier, 'actualweak1.pkl')