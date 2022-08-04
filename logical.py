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
bail = pd.read_csv('C:/Users/vedika barde/Desktop/BE_pro/bail_v.csv')
gender=1
injury=0
coaccused=0
surety=0
intension=0
eye=1
history=0
parity=1
ns=0
s=1
vs=0
period=0
difference=0
mental=0
accused=1
proof=0
delay=0
X_new=[[gender,injury,coaccused,surety,intension,eye,history,parity,ns,s,vs,period,difference,mental,accused,proof,delay]]
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

"""
x=fbail.values
min_max_scaler=preprocessing.MinMaxScaler()
x_scaled=min_max_scaler.fit_transform(x)
fbail1=pd.DataFrame(x_scaled)
"""


#All features
X = fbail.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]].values
Y = fbail.iloc[:, 17].values


#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)

#feature scaling
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)

"""
#outliers
outlier_fraction = 0.01
# store outliers and inliers in different numpy arrays
x_outliers, x_inliers = get_outliers_inliers(X_train,Y_train)
n_inliers = len(x_inliers)
n_outliers = len(x_outliers)
#separate the two features and use it to plot the data 
F1 = X_train[:,[0]].reshape(-1,1)
F2 = X_train[:,[1]].reshape(-1,1)
# create a meshgrid 
xx , yy = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))

# scatter plot 
plt.scatter(F1,F2)
plt.xlabel('F1')
plt.ylabel('F2') 
"""
#pca
pca = PCA(n_components = 2) 
X_train = pca.fit_transform(X_train) 
X_test = pca.transform(X_test) 

explained_variance = pca.explained_variance_ratio_
"""
xa=[]
xb=[]
fig, ax = plt.subplots(figsize=(16,8))
xx=np.ravel(X_train)
for i in range (len(xx)):
    if((i%2)==0):
         xa.append(xx[i])
    else:
        xb.append(xx[i])
ax.scatter(xa,xb)
ax.set_xlabel('Proportion of non-retail business acres per town')
ax.set_ylabel('Full-value property-tax rate per $10,000')
plt.show()
"""
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(fbail))
threshold = 3
print(np.where(z > 3))
fbail_1= fbail[(z < 3).all(axis=1)]
print(fbail.shape)
print(fbail_1.shape)

X = fbail_1.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]].values
Y = fbail_1.iloc[:, 17].values



from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)
"""
#naive bayes
from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB().fit(X_train, Y_train) 
Y_pred = model.predict(X_test)
print (metrics.accuracy_score(Y_test, Y_pred))
from sklearn.cross_validation import KFold
kf = KFold(25, n_folds=10, shuffle=False)

print('{} {:^61} {}'.format('Iteration', 'Training set obsevations', 'Testing set observations'))
for iteration, data in enumerate(kf, start=1):
    print('{!s:^9} {} {!s:^25}'.format(iteration, data[0], data[1]))
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(model, X, Y, cv=10, scoring='accuracy')
#print(scores)
print("NAIVE BAYES")
print(scores.mean())
X_new=pca.transform(X_new)
y_pred1=model.predict(X_new)
print(y_pred1)

#LR
#LR
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)
#predicting the test result
Y_pred2 = classifier.predict(X_test)
scores = cross_val_score(model, X, Y, cv=10, scoring='accuracy')
#print(scores)
print("LR")
y_pred1=classifier.predict(X_new)
print(y_pred1)
"""

sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)

pca = PCA(n_components = 2) 
X_train = pca.fit_transform(X_train) 
X_test = pca.transform(X_test) 

explained_variance = pca.explained_variance_ratio_
#SVM
from sklearn.svm import SVC
classifier=SVC(kernel="sigmoid",C=2,gamma=1,random_state=0, probability=True)
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
"""
# gets a dictionary of {'class_name': probability}
prob_per_class_dictionary = dict(zip(classifier.classes_, results))

# gets a list of ['most_probable_class', 'second_most_probable_class', ..., 'least_class']
results_ordered_by_probability = map(lambda x: x[0], sorted(zip(classifier.classes_, results), key=lambda x: x[1], reverse=True))
# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
#model.score(X, Y)
#Predict Output
#predicted= model.predict(X_test)

#feature importance
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_) #use inbuilt class 

import seaborn as sns
corrmat = fbail.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(fbail[top_corr_features].corr(),annot=True,cmap="RdYlGn")
"""