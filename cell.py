# -*- coding: utf-8 -*-
"""
Created on Sat May  1 10:41:50 2021

@author: thangasami
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from yellowbrick.classifier import ConfusionMatrix
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('Cellphone.csv')
X = data.drop('price_range' , 1)
y = data['price_range']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 14)

svm = SVC()
svm.fit(x_train, y_train)
score_before_scaling = svm.score(x_test, y_test)

#scaling
X = (X - np.min(X))/(np.max(X) - np.min(X))

sxtrain, sxtest, sytrain, sytest = train_test_split(X, y, test_size = .25, random_state = 14)

svm = SVC()
svm.fit(sxtrain, sytrain)
score_after_scaling = svm.score(sxtest, sytest)

ro = RobustScaler()
rx = ro.fit_transform(X)


rxtrain, rxtest, rytrain, rytest = train_test_split(rx, y, test_size = .25, random_state = 14)

svm = SVC()
svm.fit(rxtrain, rytrain)
score_robust_scaling = svm.score(rxtest, rytest)

ss = StandardScaler()
rx = ss.fit_transform(X)


rxtrain, rxtest, rytrain, rytest = train_test_split(rx, y, test_size = .25, random_state = 14)

svm = SVC()
svm.fit(rxtrain, rytrain)
score_ss_scaling = svm.score(rxtest, rytest)

train_accuracy = []
k = np.arange(1, 21)

for i in k:
    select = SelectKBest(f_classif, k=i)
    x_train_new = select.fit_transform(x_train, y_train)
    svm.fit(x_train_new, y_train)
    train_accuracy.append(svm.score(x_train_new, y_train))
    
plt.plot(k, train_accuracy, color = 'red',  label = 'Train')
plt.xlabel('k values')
plt.ylabel('Train accuracy')
plt.legend()
plt.show()

select_top = SelectKBest(f_classif, k = 4)
x_train_new = select_top.fit_transform(x_train, y_train)
x_test_new = select_top.fit_transform(x_test, y_test)
print('Top train features', x_train.columns.values[select_top.get_support()])
print('Top test features', x_test.columns.values[select_top.get_support()])

select_top = SelectKBest(f_classif, k = 6)
x_train_new = select_top.fit_transform(x_train, y_train)
x_test_new = select_top.fit_transform(x_test, y_test)
print('Top train features', x_train.columns.values[select_top.get_support()])
print('Top test features', x_test.columns.values[select_top.get_support()])


c = [1.0, 0.25, 0.5, 0.75]
kernels = ['linear', 'rbf', 'poly']
gammas = ['auto', 0.01, 0.001, 1 ]


svm = SVC()

grid_svm = GridSearchCV(estimator = svm, param_grid = dict(kernel = kernels, C = c, gamma = gammas), cv = 2)


grid_svm.fit(x_train_new, y_train)
print('The best hyperparameters are: ' , grid_svm.best_estimator_)


svc_model = SVC( C= 0.4, gamma= 'auto', kernel = 'linear')
svc_model.fit(x_train_new, y_train)

print('The train accuracy', svc_model.score(x_train_new, y_train))
print('The test accuracy', svc_model.score(x_test_new, y_test))



y_pred = svc_model.predict(x_test_new)
y_pred1 = svc_model.predict(x_train_new)

confusion_matrix(y_train, y_pred1)

#Confusion matrix
cm = ConfusionMatrix(svc_model, classes=[0,1,2,3])
cm.fit(x_train_new, y_train)
cm.score(x_train_new, y_train)
cm.show()
accuracy_score(y_test, y_pred)



















