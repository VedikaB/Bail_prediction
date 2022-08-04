# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:07:29 2019

@author: vedika barde
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
bail = pd.read_csv('C:/Users/vedika barde/Desktop/BE_pro/bail_v.csv')
from sklearn.preprocessing import LabelEncoder
number=LabelEncoder()
bail['Gender']= number.fit_transform(bail['Gender'])
bail['Output']= number.fit_transform(bail['Output'])
X = bail.iloc[:, [3,5,10,12,13,16,17,18]].values
Y = bail.iloc[:, 20].values
from scipy import stats
z = np.abs(stats.zscore(X))
threshold = 3
print(np.where(z > 3))
fbail_1= bail[(z < 3).all(axis=1)]
print(bail.shape)
print(fbail_1.shape)
#All features
X = fbail_1.iloc[:, [3,5,10,12,13,16,17,18]].values
Y = fbail_1.iloc[:, 20].values
x = StandardScaler().fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)

#feature scaling
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()
#adding the input layer and first hidden layer
classifier.add(Dense(output_dim=4,init='uniform',activation='sigmoid',input_dim=8))
#add second hidden layer
classifier.add(Dense(output_dim=4,init='uniform',activation='sigmoid'))

#add output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
y_pred=classifier.predict(X_test)
classifier.fit(X_train,Y_train,batch_size=10,nb_epoch=100)
y_pred=(y_pred>0.5)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)
































"""
pca =KernelPCA(n_components = 2, kernel="poly",degree=3, gamma=0.125 , fit_inverse_transform=True)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, bail[['logical output']]], axis = 1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [1,0]
colors = ['g', 'r']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['logical output'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
"""