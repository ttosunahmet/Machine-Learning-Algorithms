#%% Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Create Dataset

from sklearn.datasets import load_iris

iris = load_iris()

x = iris.data
y = iris.target

#%% Normalization

x = (x-np.min(x)) / (np.max(x) - np.min(x))

#%% Train Test Split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

#%% KNN Model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors= 3) # k = n_neighbors

#%% K Fold CV K = 10

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator= knn, X=x_train, y= y_train, cv= 10)
print("average accuracy : ", np.mean(accuracies))
print("acerage std : ", np.std(accuracies))

#%% KNN Fit and Accuracy

knn.fit(x_train,y_train)
print("test accuracy : ", knn.score(x_test,y_test))

#%% Grid search cross validation for knn

from sklearn.model_selection import GridSearchCV

grid = {"n_neighbors":np.arange(1,50)}
knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, grid, cv = 10)
knn_cv.fit(x,y)

#%% Print hypermater KNN 
print("tuned hyperparameter K : ", knn_cv.best_params_)
print("tuned parametreye gore en iyi accuracy (best score) : ", knn_cv.best_score_)

#%% Grid search CV with Logistic Regression

x = x[:100,:]
y = y[:100]

from sklearn.linear_model import LogisticRegression

grid = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"], "solver": ["liblinear", "saga"]} # l1 = lasso , l2 = ridge

logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(x,y)

print("tuned hyperparameters (best parameters) : ", logreg_cv.best_params_)
print("accuracy : ", logreg_cv.best_score_)