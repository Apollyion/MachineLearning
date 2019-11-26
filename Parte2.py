# -*- coding: utf-8 -*-
"""
Created on Fry Nov 2 22:15:05s 2019

@author: lucas
"""

from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.svm import NuSVC
from sklearn.ensemble import RandomForestClassifier


dados = load_breast_cancer()

#Dividir Dataset
train, test, train_labels, test_labels = train_test_split(dados['data'], dados['target'], test_size=0.3, random_state=0)

#Alg 1 - GNB
gnb = GaussianNB()  
gnb.fit(train, train_labels)
predicted = gnb.predict(test)

print("Relatório do classificador GaussianNB \n  %s:\n%s\n" % (gnb, metrics.classification_report(test_labels, predicted)))

#Alg 2 - KMeans
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(train, train_labels)
predicted1 = kmeans.predict(test)
print("Relatório do classificador KMeans \n  %s:\n%s\n" % (KMeans, metrics.classification_report(test_labels, predicted1)))

#Alg 3 - SVC
clf = NuSVC(gamma='scale')
clf.fit(train, train_labels)
predicted2 = clf.predict(test)
print("Relatório do classificador NuSVC \n  %s:\n%s\n" % (clf, metrics.classification_report(test_labels, predicted2)))

#Alg 4 - RFC
classifier = RandomForestClassifier(n_estimators=10)
classifier.fit(train, train_labels)
predicted3 = classifier.predict(test)
print("Relatório do RandomForest \n  %s:\n%s\n" % (classifier, metrics.classification_report(test_labels, predicted3)))
print(dados['DESCR'])
