#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
from sklearn import svm
from sklearn.metrics import accuracy_score
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
#clf = svm.SVC(kernel='linear')
clf = svm.SVC(kernel='rbf', C=10000.0) # accuracy C is 10.0 or 100.0 = same as default C, improves to .82 for C=1000, for C=1000 to .89
t1 = time()

#Reducing the training set to shorten the training time, but generally the accuracy is hit by doing this
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

clf.fit(features_train, labels_train)
print 'training time: ', time()-t1 #180.788 seconds

t1 = time()
pred = clf.predict(features_test)

#print 'prediction for 10th entry:',pred[10],',', 'prediction for 26th entry:',pred[26],'and', 'prediction for 50th entry:',pred[50]
print 'prediction time: ', time()-t1 #18.09 seconds
print accuracy_score(pred, labels_test) #Accuracy achieved with linear kernel: 0.984, rbf kernel with default C: 0.616

count = 0
for x in pred:
    if x == 1:
        count+1

print 'No of Chris Emails in test:', count
#########################################################


