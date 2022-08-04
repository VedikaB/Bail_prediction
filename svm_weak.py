# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 17:10:59 2019

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
bail = pd.read_csv('C:/Users/vedika barde/Desktop/BE_pro/Bailfinal.csv')
gender=1
coaccused=0
period=0
difference=0
accused=1
delay=0
X_new=[[gender,coaccused,period,difference,accused,delay]]
#Encoding
from sklearn.preprocessing import LabelEncoder
number=LabelEncoder()
bail['Gender']= number.fit_transform(bail['Gender']) #bail[['Nature of Alligations']]
# transform and map pokemon generations
#gen_labels= number.fit_transform(bail['Nature of Alligations'].astype('str'))
#bail['Gen_Label'] = gen_labels 
# transform and map pokemon legendary status


#All features
X = bail.iloc[:, [3,5,12,13,16,18]].values
Y = bail.iloc[:, 19].values


#train test split
from sklearn.model_selection import train_test_split
X_train3, X_test3, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)

#feature scaling
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train3)
X_test=sc_x.transform(X_test3)

#pca
pca = PCA(n_components = 2) 
X_train = pca.fit_transform(X_train) 
X_test = pca.transform(X_test) 

explained_variance = pca.explained_variance_ratio_

#SVM
from sklearn.svm import SVC
classifier=SVC(kernel="poly",random_state=0, probability=True)
classifier.fit(X_train,Y_train)
from sklearn.cross_validation import KFold
kf = KFold(25, n_folds=10, shuffle=False)

print('{} {:^61} {}'.format('Iteration', 'Training set obsevations', 'Testing set observations'))
for iteration, data in enumerate(kf, start=1):
    print('{!s:^9} {} {!s:^25}'.format(iteration, data[0], data[1]))
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(classifier, X, Y, cv=10, scoring='accuracy')
print("SVM")
print(scores)
print(scores.mean())

X_new=pca.transform(X_new)
y_pred1=classifier.predict(X_new)
print(y_pred1)
results = classifier.predict_proba(X_new)[0]
for i in range(len(X_new)):
    if((results[0])>0.5):
        print(results[0])
    else:
        print(results[1]) 
#plotting

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

plt.title('SVM (Training set)') 
plt.xlabel('PC1') # for Xlabel 
plt.ylabel('PC2') # for Ylabel 
plt.legend() # to show legend 
# show scatter plot 
plt.show()
from sklearn.externals import joblib
joblib.dump(classifier, 'actualweaksvm.pkl')
