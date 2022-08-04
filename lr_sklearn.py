# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:53:05 2019

@author: vedika barde
"""
import matplotlib.pyplot as plot
import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
dataset = pandas.read_csv('lr.csv')
x = dataset.iloc[:, [0]].values
y = dataset.iloc[:, 1].values

# Split the dataset into the training set and test set
# We're splitting the data in 1/3, so out of 30 rows, 20 rows will go into the training set,
# and 10 rows will go into the testing set.
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 1/3, random_state = 0)
linearRegressor = LinearRegression()
linearRegressor.fit(x, y)
print(linearRegressor.coef_[0])
print(linearRegressor.intercept_)
yPrediction = linearRegressor.predict(xTest)
plot.scatter(x, y, color = 'red')
plot.plot(x,linearRegressor.predict(x), color = 'blue')
plot.title('Salary vs Experience (Training set)')
plot.xlabel('Years of Experience')
plot.ylabel('Salary')
plot.show()