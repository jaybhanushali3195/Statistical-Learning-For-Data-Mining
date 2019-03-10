# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 00:59:17 2018

@author: VIREN
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 11:41:39 2018

@author: runger
"""

#import matplotlib.pyplot as plt
#import numpy as np
import pandas as pd
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from pandas_ml import ConfusionMatrix
from sklearn import model_selection

f = open('winequality-red.csv')
csv_f = csv.reader(f)
attendees1 = []
for row in csv_f:
    print (row)
f.close()
mydata = pd.read_csv('winequality-red.csv', index_col=0)
columns = " fixed acidity	volatile acidity	citric acid	residual sugar	chlorides	free sulfur dioxide	total sulfur dioxide	density	pH	sulphates	alcohol	quality".split()
df = pd.DataFrame(mydata.data, columns = columns)
#x1=pd.bank.csv
#mydata= datasets.load_bank()

#mydata.DESCR

X=mydata.train
y=mydata.target
seedMLP = 2357 #rANDOM nUMBER ASSIGNED
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

# Next line is a neural network, but any model can be used
modelnow=MLPClassifier(hidden_layer_sizes=(20,20,20), random_state=seedMLP)
#modelnow = GaussianNB()
modelnow.fit(X,y)

# Compute training error
yhat = modelnow.predict(X)
print ("Training Error")
print ("Error")
print (metrics.accuracy_score(y, yhat))
print (metrics.classification_report(y, yhat))
print (ConfusionMatrix(y, yhat))
#np.random.seed()
seed=719
actuals=[]
probs=[]
hats=[]

kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)
for train, test in kfold.split(X, y):
    #print('train: %s, test: %s' % (train, test))
    # Train classifier on training data, predict test data
    modelnow.fit(X[train], y[train])
    foldhats = modelnow.predict(X[test])
    foldprobs = modelnow.predict_proba(X[test])[:,1] # Class probability estimates for ROC curve
    actuals = np.append(actuals, y[test]) #Combine targets, then probs, and then predictions from each fold
    probs = np.append(probs, foldprobs)
    hats = np.append(hats, foldhats)

print ("Crossvalidation Error")    
print ("CVerror = ", metrics.accuracy_score(actuals,hats))
print (metrics.classification_report(actuals, hats))
cm = ConfusionMatrix(actuals,hats)
print (cm)
cm.print_stats()


if len(mydata.target_names) == 2: #ROC curve code here only for 2 classes
    fpr, tpr, threshold = metrics.roc_curve(actuals, probs)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic ')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
