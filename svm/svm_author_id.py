#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn import svm
from sklearn.metrics import accuracy_score

# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

cls = svm.SVC(kernel="rbf", C=10000)

t0 = time()
cls.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"


t0 = time()
pred = cls.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"

# print "prediction for 10th: ", "Sara" if pred[10] == 0 else "Chris"
# print "prediction for 26th: ", "Sara" if pred[26] == 0 else "Chris"
# print "prediction for 50th: ", "Sara" if pred[50] == 0 else "Chris"

match1 = 0
match0 = 0
for e in pred:
    if e == 1:
        match1 += 1
    else:
        match0 += 1

print "Sara mails: ", str(match0)
print "Chris mails: ", str(match1)

accuracy = accuracy_score(labels_test, pred)
print "accuracy: ", accuracy

#########################################################


