
"""
Code IDE is spyder
Using logistic regression in scikit to iris dataset

"""
# read the iris data set
from sklearn.datasets import load_iris
iris=load_iris()

# create X(featuures) and y(response)
X=iris.data
y=iris.target

# importing the logistic regression class
from sklearn.linear_model import LogisticRegression

# assigning a variable to a class
log_reg=LogisticRegression()

# fit the model with data
log_reg.fit(X, y) 

# predict the response values for the observations in X
log_reg.predict(X)

# store the predicted response value
y_pred = log_reg.predict(X)

# now to check accuracy of the applied logistic regession class
from sklearn import metrics
print (metrics.accuracy_score( y, y_pred))
