# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 16:11:16 2019

@author: vedika barde
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
bail = pd.read_csv('C:/Users/vedika barde/Desktop/BE_pro/Bailfinal.csv')
#Encoding
from sklearn.preprocessing import LabelEncoder
number=LabelEncoder()
bail['Gender']= number.fit_transform(bail['Gender'].astype('str'))
bail['Mental codition of accused']= number.fit_transform(bail['Mental codition of accused'].astype('str'))
bail[['Nature of Alligations']]
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# transform and map pokemon generations
gen_labels= number.fit_transform(bail['Nature of Alligations'].astype('str'))
bail['Gen_Label'] = gen_labels 
# transform and map pokemon legendary status
poke_df_sub = bail[['Gender','Injuiry','co-accused','Surety of Bond','Intention','Eyewitness','History','Law of Parity']]
poke_df_sub1= bail[['Period of custody','Differences b/w chargesheet and trials','Mental codition of accused','Loophole of investigation','Accused presented within 24 hrs','Proof with accused','Delay in launching FIR/producing chargesheet','logical output','Output']]
# encode generation labels using one-hot encoding scheme
gen_ohe = OneHotEncoder()
gen_feature_arr = gen_ohe.fit_transform(
                              bail[['Gen_Label']]).toarray()
gen_feature_labels = list(number.classes_)
gen_features = pd.DataFrame(gen_feature_arr, 
                            columns=gen_feature_labels)
fbail = pd.concat([poke_df_sub, gen_features, poke_df_sub1], axis=1)
columns = sum([['Gender','Injuiry','co-accused','Surety of Bond','Intention','Eyewitness','History','Law of Parity'],gen_feature_labels] , [])
fbail[columns]
fbail['Output']= number.fit_transform(bail['Output'].astype('str'))
#All features
X = fbail.iloc[:, [1,3,4,5,6,7,8,9,10,13,14,16]].values
Y = fbail.iloc[:, 19].values
#feature scaling
sc_x=StandardScaler()
X=sc_x.fit_transform(X)    
#pca
pca = PCA(n_components = 2) 
X = pca.fit_transform(X) 
explained_variance = pca.explained_variance_ratio_
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(X))
threshold = 3
print(np.where(z > 3))
fbail_1= fbail[(z < 3).all(axis=1)]
print(fbail.shape)
print(fbail_1.shape)
    
X = fbail_1.iloc[:, [1,3,4,5,6,7,8,9,10,13,14,16]].values
Y = fbail_1.iloc[:, 19].values
sc_x=StandardScaler()
X=sc_x.fit_transform(X)
pca = PCA(n_components = 2) 
from sklearn.model_selection import train_test_split
X_train2, X_test2, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)
X_train = pca.fit_transform(X_train2) 
X_test = pca.transform(X_test2) 
explained_variance = pca.explained_variance_ratio_
#SVM
from sklearn.svm import SVC
classifier_s=SVC(kernel="poly",degree=10,random_state=0, probability=True)
classifier_s.fit(X_train,Y_train)
from sklearn.cross_validation import KFold
kf = KFold(25, n_folds=10, shuffle=False)
    
print('{} {:^61} {}'.format('Iteration', 'Training set obsevations', 'Testing set observations'))
for iteration, data in enumerate(kf, start=1):
    print('{!s:^9} {} {!s:^25}'.format(iteration, data[0], data[1]))
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(classifier_s, X, Y, cv=10, scoring='accuracy')
print("SVM")
print(scores)
print(scores.mean())
#plotting
from matplotlib.colors import ListedColormap 
X_set, y_set = X_train, Y_train 
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, 
					stop = X_set[:, 0].max() + 1, step = 0.01), 
					np.arange(start = X_set[:, 1].min() - 1, 
					stop = X_set[:, 1].max() + 1, step = 0.01)) 
    
plt.contourf(X1, X2, classifier_s.predict(np.array([X1.ravel(), 
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

joblib.dump(classifier_s, 'actualstrong.pkl')
