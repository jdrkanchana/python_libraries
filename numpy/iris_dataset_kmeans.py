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

# dataset split ,0.6 ratio train data set, 0.4 ratio test data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4)

# import the kmeans class in scikit learn
from sklearn.neighbors import KNeighborsClassifier

# introduce the no. of neighbors, first try putting 5 neighbours in cluster then try other neighbor values
# introduced an ohect, a shortform to call the k-means class
knn =KNeighborsClassifier(n_neighbors=5)

# fit the model with data
knn.fit(X_train, y_train)

# predicting response values for the observation
y_pred=knn.predict(X_test)

# check the accuracy when n=5 using k-means on the dataset
from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred))

# introduced 1 as neighbor cluster
knn =KNeighborsClassifier(n_neighbors=1)

# fit the model with data
knn.fit(X_train, y_train)

# predicting response values for the observation
y_pred=knn.predict(X_test)

# check the accuracy when n=1 using k-means on the dataset
from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred))

# to find using a for loop the best value n for knn can take

# from k=1 to k=19
k_range=range(1,20)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred=knn.predict(X_test)
    scores.append(metrics.accuracy_score( y_test, y_pred))

# out of all n values in knn the maximum accuracy of the model used 
max_accuracy=max(scores)
print("the maximum accuracy of the predicted response value",max_accuracy)
print("The neighbour which gives the maximum accuracy",scores.index(max_accuracy))
# learning outcome, per each time code executed the train and test data set split method vary
# hence accuracy of model varies, hence at given point the neighbour suggested varies
