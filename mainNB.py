# -*- coding: utf-8 -*-
"""
Created on Fri May 17 23:28:50 2019

@author: Kevin Kusumah
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Visit-Nominal.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


#Encoding Data
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
X[:,1] = labelencoder_X.fit_transform(X[:,1])
X[:,2] = labelencoder_X.fit_transform(X[:,2])
X[:,3] = labelencoder_X.fit_transform(X[:,3])
X[:,4] = labelencoder_X.fit_transform(X[:,4])
X[:,5] = labelencoder_X.fit_transform(X[:,5])

y = labelencoder_X.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Fitting Classifier to Training Set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn import metrics
print("Accuracy Model: ",metrics.accuracy_score(y_test, y_pred))

##Cross Validation
#from sklearn.model_selection import cross_val_score
#cv = cross_val_score(classifier, X,y, cv=10)
#print("Accuracy 10-Cross Validation : ", cv.mean())


##Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
#print('Confusion Matrix :\n',cm)

#labels = ['Class 0','Class 1']
#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(cm, cmap=plt.cm.Blues)
#fig.colorbar(cax)
#ax.set_xticklabels([''] + labels)
#ax.set_yticklabels([''] + labels)
#plt.xlabel('Predicted')
#plt.ylabel('Expected')
#plt.show()
