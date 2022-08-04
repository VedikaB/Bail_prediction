# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 14:32:13 2018

@author: vedika barde
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('C:/Users/vedika barde/Desktop/BE_pro/bail_final.csv')
X = dataset.iloc[:, [8,9]].values
Y = dataset.iloc[:, [10]].values
# Spliting the data set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
Y_train=Y_train.ravel();
"""
ph = 3
alcohol = 10
X_new=[[ph,alcohol]]
#Y_test=Y_test.ravel();
"""
#feature scaling
#from sklearn.preprocessing import StandardScaler
#sc_X=StandardScaler()
#X_train=sc_X.fit_transform(X_train)
#X_test=sc_X.transform(X_test)

 
# KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski' , p = 2)
classifier.fit(X_train , Y_train)
#Prediction

Y_pred = classifier.predict(X_test)
from sklearn import metrics
metrics.accuracy_score(Y_test, Y_pred)
Y_pred_k = classifier.predict(X_new)

#Y_pred = classifier.predict(X_test)
#confusion matrix
"""
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test , Y_pred)
cm_d=cm.ravel()
print(cm[1])
#plot_confusion_matrix(cm)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred_d = classifier.predict(X_new)

Y_pred1 = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(Y_test , Y_pred1)


#fitting logistic regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

#predicting the test result
Y_pred_L = classifier.predict(X_new)

Y_pred2 = classifier.predict(X_test)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(Y_test, Y_pred2)











X = dataset.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,13,14]].values
Y = dataset.iloc[:, [12]].values


# Spliting the data set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB().fit(X_train, Y_train) 
Y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test , Y_pred)
cm_d=cm.ravel()
print(cm)
"""