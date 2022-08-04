# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 19:45:14 2019

@author: vedika barde
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 10:54:37 2019

@author: vedika barde
"""
# importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

dataset = pd.read_csv('C:/Users/vedika barde/Desktop/BE_pro/bail_v.csv')

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
number=LabelEncoder()
dataset['Gender']= number.fit_transform(dataset['Gender'].astype('str'))
dataset['Mental codition of accused']= number.fit_transform(dataset['Mental codition of accused'].astype('str'))
dataset['Nature of Alligations']= number.fit_transform(dataset['Nature of Alligations'].astype('str'))


dataset['Output']= number.fit_transform(dataset['Output'].astype('str'))

# Spliting the data set
from sklearn.cross_validation import train_test_split
X = dataset.iloc[:, [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]].values
Y = dataset.iloc[:, 20].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

#ohe = OneHotEncoder(categorical_features = [8]) 
#label_encoded_data = number.fit_transform(ohe)
#ohe.fit_transform(label_encoded_data.reshape(-1,1))

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
print(scores)
print("NAIVE BAYES")
print(scores.mean())
pca = sklearnPCA(n_components=2) #2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(X))
#
plt.scatter(transformed[Y==0][0], transformed[Y==0][1], label='Granted', c='blue')
plt.scatter(transformed[Y==1][0], transformed[Y==1][1], label='Not Granted', c='red')
plt.legend()
plt.show()

"""
"""
pos , neg = (Y==1).reshape(225,1) , (Y==0).reshape(225,1)
plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+")
plt.scatter(X[neg[:,0],0],X[neg[:,0],1],marker="o",s=10)
plt.xlabel("Test 1")
plt.ylabel("Test 2")
plt.legend(["Accepted","Rejected"],loc=0)
"""
"""
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test , Y_pred)
#print(cm)



from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

#predicting the test result
Y_pred2 = classifier.predict(X_test)
scores = cross_val_score(classifier, X, Y, cv=10, scoring='accuracy')
print(scores)
print("LR")
print(scores.mean())
#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(Y_test, Y_pred2)
#print(cm2)
#from sklearn.cross_validation import cross_val_score
#scores = cross_val_score(model, X, Y, cv=10, scoring='accuracy')
#print(scores)



# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results

Y_pred1 = classifier.predict(X_test)
scores = cross_val_score(classifier, X, Y, cv=10, scoring='accuracy')
print(scores)
print("DECISION TREE")
print(scores.mean())
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(Y_test , Y_pred1)

#Import Library

#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object 
model = svm.SVC(kernel='sigmoid', C=1, gamma='auto', degree=3) 
# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
model.fit(X, Y)
model.score(X, Y)
#Predict Output
predicted= model.predict(X_test)
Y_pred1 = classifier.predict(X_test)
scores = cross_val_score(classifier, X, Y, cv=10, scoring='accuracy')
print(scores)
print("SVM")
print(scores.mean())





# predicting with random forest
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)  
regressor.fit(X_train, Y_train)  
y_pred = regressor.predict(X_test)  
scores = cross_val_score(regressor, X, Y, cv=10, scoring='accuracy')
#print(scores)
print("RF")
print(scores.mean())
