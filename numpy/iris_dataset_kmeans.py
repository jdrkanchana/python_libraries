"""
Code IDE is spyder
Using k-means in scikit to iris dataset

@author: dul
"""
# read in the iris data set
from sklearn.datasets import load_iris
iris = load_iris()

# create X(featues/attributes) and y (response)
X=iris.data
y=iris.target

# import the kmeans class in scikit learn
from sklearn.neighbors import KNeighborsClassifier

# introduce the no. of neighbors, first try putting 5 neighbours in cluster then try other neighbor values
# introduced an ohect, a shortform to call the k-means class
knn =KNeighborsClassifier(n_neighbors=5)

# fit the model with data
knn.fit(X, y)

# predicting response values for the observation
y_pred=knn.predict(X)

# check the accuracy when n=5 using k-means on the dataset
from sklearn import metrics
print(metrics.accuracy_score(y,y_pred))

# introduced 1 as neighbor cluster
knn =KNeighborsClassifier(n_neighbors=1)

# fit the model with data
knn.fit(X, y)

# predicting response values for the observation
y_pred=knn.predict(X)

# check the accuracy when n=1 using k-means on the dataset
from sklearn import metrics
print(metrics.accuracy_score(y,y_pred))
