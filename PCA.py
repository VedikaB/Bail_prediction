# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:47:38 2019

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
bail = pd.read_csv('C:/Users/vedika barde/Desktop/BE_pro/bail_v.csv')
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
poke_df_sub1= bail[['Period of custody','Differences b/w chargesheet and trials','Mental codition of accused','Accused presented within 24 hrs','Proof with accused','Delay in launching FIR/producing chargesheet','logical output']]
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

X = fbail.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]].values
Y = fbail.iloc[:, 17].values
# Standardizing the features
x = StandardScaler().fit_transform(X)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, fbail[['logical output']]], axis = 1)
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