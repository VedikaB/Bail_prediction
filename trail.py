# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 22:26:51 2019

@author: vedika barde
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
 
 # instantiating 
bail = pd.read_csv('C:/Users/vedika barde/Desktop/BE_pro/bail_v.csv')
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
number=LabelEncoder()
bail['Gender']= number.fit_transform(bail['Gender'].astype('str'))
bail['Mental codition of accused']= number.fit_transform(bail['Mental codition of accused'].astype('str'))
bail['Nature of Alligations']= number.fit_transform(bail['Nature of Alligations'].astype('str'))
bail['Output']= number.fit_transform(bail['Output'].astype('str'))

X = bail.iloc[:, [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]].values
Y = bail.iloc[:, 20].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)
# Applying PCA function on training 
# and testing set of X component 

from sklearn.decomposition import PCA 

pca = PCA(n_components = 2) 

X_train = pca.fit_transform(X_train) 
X_test = pca.transform(X_test) 

explained_variance = pca.explained_variance_ratio_ 

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

#predicting the test result
Y_pred2 = classifier.predict(X_test)
from matplotlib.colors import ListedColormap 

X_set, y_set = X_train, Y_train 
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, 
					stop = X_set[:, 0].max() + 1, step = 0.01), 
					np.arange(start = X_set[:, 1].min() - 1, 
					stop = X_set[:, 1].max() + 1, step = 0.01)) 

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), 
			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
			cmap = ListedColormap(('yellow', 'white'))) 

plt.xlim(X1.min(), X1.max()) 
plt.ylim(X2.min(), X2.max()) 

for i, j in enumerate(np.unique(y_set)): 
	plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
				c = ListedColormap(('red', 'green'))(i), label = j) 

plt.title('Logistic Regression (Training set)') 
plt.xlabel('PC1') # for Xlabel 
plt.ylabel('PC2') # for Ylabel 
plt.legend() # to show legend 

# show scatter plot 
plt.show() 



