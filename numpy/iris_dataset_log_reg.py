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

# dataset split ,0.6 ratio train data set, 0.4 ratio test data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4)

# importing the logistic regression class
from sklearn.linear_model import LogisticRegression

# assigning a variable to a class
log_reg=LogisticRegression()

# fit the model with data
log_reg.fit(X_train, y_train) 

# predict the response values for the observations in X
log_reg.predict(X_test)

# store the predicted response value
y_pred = log_reg.predict(X_test)

# now to check accuracy of the applied logistic regession class
from sklearn import metrics
print (metrics.accuracy_score( y_test, y_pred))
