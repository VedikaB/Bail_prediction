import pandas as pd

data = pd.read_csv('knn.csv')

from sklearn.neighbors import KNeighborsClassifier

x = data.values[:,0:2]
y = data.values[:,2]
print(x)
print(y)

#Basic KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x, y) 
print(model.predict([[4,4]]))


#distance Weighted KNN
weightedknn = KNeighborsClassifier(n_neighbors=5, weights='distance')
weightedknn.fit(x,y)
print(weightedknn.predict([[6,2]]))


#average Weighted KNN
distarray,indarray = weightedknn.kneighbors([[6,2]])
print(distarray)
print(indarray)
pos=0
neg=0
poscount=0
negcount=0
for val in indarray:
	for i in range(len(val)):
		if y[i]=='positive':
			poscount+=1
			pos=pos+distarray[0][i]
		else:
			negcount+=1
			neg=neg+distarray[0][i]

pos=float(pos/poscount)
neg=float(neg/negcount)
if pos>neg:
	print ("Class is positive")
else:
	print ("class in negative")

