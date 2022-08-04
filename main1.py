# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:17:28 2019

@author: vedika barde
"""

from logicalmodel import classifier_svm
from logicalmodel import pca,X_train1,X_test1,sc_x
from actualstrong import X_train2,X_test2
from actualstrong import classifier_s
from svm_weak import X_train3,X_test3
from svm_weak import classifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

gender=1
injury=1
coaccused=0
surety=0
intension=1
eye=1
history=1
parity=0
ns=0
s=0
vs=1
period=0
difference=0
mental=0
accused=1
proof=0
delay=0
loopholes=1
X_new=[[gender,injury,coaccused,surety,intension,eye,history,parity,ns,s,vs,period,difference,mental,accused,proof,delay]] 
X_train=sc_x.fit_transform(X_train1) 
X_test=sc_x.transform(X_test1) 
X_new=sc_x.transform(X_new)  
X_train = pca.fit_transform(X_train) 
X_test = pca.transform(X_test) 
X_new=pca.transform(X_new)
explained_variance = pca.explained_variance_ratio_
y_predl=classifier_svm.predict(X_new)
results = classifier_svm.predict_proba(X_new)[0]
for i in range(len(X_new)):
    if((results[0])>0.5):
        print(results[0])
    else:
        print(results[1])
if(y_predl==1):
    print("Bail Granted")
else:
    X_news=[[injury,surety,intension,eye,history,parity,ns,s,vs,mental,loopholes,proof]]
    X_trains=sc_x.fit_transform(X_train2) 
    X_tests=sc_x.transform(X_test2) 
    X_news=sc_x.transform(X_news)  
    X_train = pca.fit_transform(X_trains) 
    X_test = pca.transform(X_tests) 
    X_news=pca.transform(X_news)
    explained_variance = pca.explained_variance_ratio_
    y_preds=classifier_svm.predict(X_news)
    print(y_preds)
    results = classifier_svm.predict_proba(X_news)[0]
    for i in range(len(X_news)):
        if((results[0])>0.5):
            a=results[0]
        else:
            a=results[1]
    print("Strong")
    print(a)

    X_neww=[[gender,coaccused,period,difference,accused,delay]]
    X_trainw=sc_x.fit_transform(X_train3) 
    X_testw=sc_x.transform(X_test3) 
    X_neww=sc_x.transform(X_neww)
    X_train = pca.fit_transform(X_trainw) 
    X_test = pca.transform(X_testw)
    X_neww=pca.transform(X_neww)
    explained_variance = pca.explained_variance_ratio_
    y_predw=classifier.predict(X_neww)
    print(y_predw)
    results = classifier.predict_proba(X_neww)[0]
    for i in range(len(X_neww)):
        if((results[0])>0.5):
            a=results[0]
        else:
            a=results[1] 
    print("Weak")
    print(a)
    
    if(y_preds==0 and y_predw==0):
        print("Bail can not be granted")
    elif(y_preds==1 and y_predw==1):
        print("Bail will be granted")
    else:
        print("Results depends on lawyers experience and intutions")
    print(a)
from sklearn.externals import joblib
joblib.dump(y_predl, 'main11.pkl')
#joblib.dump(y_preds, 'main12.pkl')
#joblib.dump(y_predw, 'main13.pkl')
