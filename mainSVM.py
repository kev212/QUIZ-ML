# -*- coding: utf-8 -*-
"""
Created on Sat May 18 19:34:53 2019

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

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#Fitting Classifier to Training Set / Kernel 'rbf' & C=0.01
from sklearn.svm import SVC
yrbf = []  #for plotting
clf_rbf1 = SVC(kernel='rbf', C=0.01, random_state = 0)

# 5-Cross Validation
from sklearn.model_selection import cross_val_score
accuracies1 = cross_val_score(clf_rbf1, X,y, cv = 5)
acc1_mean = accuracies1.mean()
yrbf.append(acc1_mean)

#Fitting Classifier to Training Set / Kernel 'rbf' & C=0.1
clf_rbf2 = SVC(kernel='rbf', C=0.1, random_state = 0)

# 5-Cross Validation
accuracies2 = cross_val_score(clf_rbf2, X, y, cv = 5)
acc2_mean = accuracies2.mean()
yrbf.append(acc2_mean)

#Fitting Classifier to Training Set / Kernel 'rbf' & C=1
clf_rbf3 = SVC(kernel='rbf', C=1, random_state = 0)

# 5-Cross Validation
accuracies3 = cross_val_score(clf_rbf3, X, y, cv = 5)
acc3_mean = accuracies3.mean()
yrbf.append(acc3_mean)

#Fitting Classifier to Training Set / Kernel 'rbf' & C=10
clf_rbf4 = SVC(kernel='rbf', C=10, random_state = 0)

# 5-Cross Validation
accuracies4 = cross_val_score(clf_rbf4, X, y, cv = 5)
acc4_mean = accuracies4.mean()
yrbf.append(acc4_mean)

#Fitting Classifier to Training Set / Kernel 'poly' & C=0.01
ypoly = [] #For Plotting
clf_poly5 = SVC(kernel='poly', C=0.01, random_state = 0)

# 5-Cross Validation
accuracies5 = cross_val_score(clf_poly5, X, y, cv = 5)
acc5_mean = accuracies5.mean()
ypoly.append(acc5_mean)

#Fitting Classifier to Training Set / Kernel 'poly' & C=0.1
clf_poly6 = SVC(kernel='poly', C=0.1, random_state = 0)

# 5-Cross Validation
accuracies6 = cross_val_score(clf_poly6, X, y, cv = 5)
acc6_mean = accuracies6.mean()
ypoly.append(acc6_mean)

#Fitting Classifier to Training Set / Kernel 'poly' & C=1
clf_poly7 = SVC(kernel='poly', C=1, random_state = 0)

# 5-Cross Validation
accuracies7 = cross_val_score(clf_poly7, X, y, cv = 5)
acc7_mean = accuracies7.mean()
ypoly.append(acc7_mean)

#Fitting Classifier to Training Set / Kernel 'poly' & C=10
clf_poly8 = SVC(kernel='poly', C=10, random_state = 0)

# 5-Cross Validation
accuracies8 = cross_val_score(clf_poly8, X, y, cv = 5)
acc8_mean = accuracies8.mean()
ypoly.append(acc8_mean)

#Fitting Classifier to Training Set / Kernel 'sigmoid' & C=0.01
ysigmoid = []
clf_sigmoid9 = SVC(kernel='sigmoid', C=0.01, random_state = 0)

# 5-Cross Validation
accuracies9 = cross_val_score(clf_sigmoid9, X, y, cv = 5)
acc9_mean = accuracies9.mean()
ysigmoid.append(acc9_mean)

#Fitting Classifier to Training Set / Kernel 'sigmoid' & C=0.1
clf_sigmoid10 = SVC(kernel='sigmoid', C=0.1, random_state = 0)

# 5-Cross Validation
accuracies10 = cross_val_score(clf_sigmoid10, X, y, cv = 5)
acc10_mean = accuracies10.mean()
ysigmoid.append(acc10_mean)

#Fitting Classifier to Training Set / Kernel 'sigmoid' & C=1
clf_sigmoid11 = SVC(kernel='sigmoid', C=1, random_state = 0)

# 5-Cross Validation
accuracies11 = cross_val_score(clf_sigmoid11, X, y, cv = 5)
acc11_mean = accuracies11.mean()
ysigmoid.append(acc11_mean)

#Fitting Classifier to Training Set / Kernel 'sigmoid' & C=10
clf_sigmoid12 = SVC(kernel='sigmoid', C=10, random_state = 0)

# 5-Cross Validation
accuracies12 = cross_val_score(clf_sigmoid12, X, y, cv = 5)
acc12_mean = accuracies12.mean()
ysigmoid.append(acc12_mean)


Xc = [0.01, 0.1, 1, 10]
#Plotting Accuracy Kernel 'rbf'
plt.xlabel('Nilai C')
plt.ylabel('Akurasi')
plt.title('Akurasi Kernel "rbf"')
plt.bar(Xc, yrbf)
plt.savefig("rbf.png")
plt.show()

#Plotting Accuracy Kernel 'poly'
plt.xlabel('Nilai C')
plt.ylabel('Akurasi')
plt.title('Akurasi Kernel "poly"')
plt.bar(Xc, ypoly)
plt.savefig("poly.png")
plt.show()

#Plotting Accuracy Kernel 'sigmoid'
plt.xlabel('Nilai C')
plt.ylabel('Akurasi')
plt.title('Akurasi Kernel "sigmoid"')
plt.bar(Xc, ysigmoid)
plt.savefig("sigmoid.png")
plt.show()


