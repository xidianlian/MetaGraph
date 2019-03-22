# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 09:30:20 2019

@author: lianWeiC
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, recall_score,classification_report
from sklearn.ensemble import RandomForestClassifier

api_csv_path = "E:\\Spyder\\android_malware_detection\\api_ben_top8000.csv"
perm_csv_path = "E:\\Spyder\\android_malware_detection\\perm.csv"
#method_csv_path = "E:\\lian_workspace\\MetaGraph\\exe\\method.csv"
app_api_data = pd.read_csv(api_csv_path, dtype=np.int8)
app_perm_data = pd.read_csv(perm_csv_path, dtype=np.int8)
#app_method_data = pd.read_csv(method_csv_path, dtype=np.int8)

label = pd.DataFrame(app_api_data['class'], columns=["class"])
app_api_data = app_api_data.drop(['class'], axis = 1)
app_perm_data = app_perm_data.drop(['class'], axis = 1)
#app_method_data = app_method_data.drop(['class'], axis = 1)
selector = SelectKBest(score_func=chi2, k=400)
new_api_data = selector.fit_transform(app_api_data, label.values.ravel())
#selector1 = SelectKBest(score_func=chi2, k=800)
#new_method_data = selector1.fit_transform(app_method_data, label.values.ravel())

#合并
new_api_data = pd.DataFrame(new_api_data)
#new_method_data = pd.DataFrame(new_method_data)
new_data =  pd.concat([new_api_data, app_perm_data], axis = 1) 

X_train, X_test, Y_train, Y_test = train_test_split(new_data, label, test_size=0.2, shuffle=False)

lr = LogisticRegression(C = 1, penalty='l1', solver="liblinear")
# 根据所有的交叉验证，计算相应的score 
scores = cross_val_score(lr, X_train, Y_train.values.ravel(), cv=5)
print("LogisticRegression the mean of score:", scores.mean())

rfc = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=5,min_samples_leaf=3)
scores = cross_val_score(rfc, X_train, Y_train.values.ravel(), cv=5)
print("RandomForestClassifier 100 & chi2 the mean of score:", scores.mean())
rfc.fit(X_train, Y_train.values.ravel())
Y_pre = rfc.predict(X_test)
cnf_matrix = confusion_matrix(Y_test, Y_pre)
recall = cnf_matrix[1,1] / (cnf_matrix[1,1] + cnf_matrix[1,0]) 
acc = (cnf_matrix[1,1] + cnf_matrix[0,0])/(cnf_matrix[1,1] + cnf_matrix[1,0]+ cnf_matrix[0,0] + cnf_matrix[0,1])
precision = cnf_matrix[1,1] / (cnf_matrix[1,1] + cnf_matrix[0,1])
F1 = 2 * precision * recall / (precision + recall) 
print('RandomForestClassifier test Recall : ', recall)
print('RandomForestClassifier test acc : ', acc)
print('RandomForestClassifier test precision ', precision)
print('RandomForestClassifier test F1 ', F1)


clf = svm.SVC(C=2, kernel='rbf', gamma='auto')
clf.fit(X_train, Y_train.values.ravel())
Y_pre = clf.predict(X_test)
cnf_matrix = confusion_matrix(Y_test, Y_pre)
recall = cnf_matrix[1,1] / (cnf_matrix[1,1] + cnf_matrix[1,0]) 
acc = (cnf_matrix[1,1] + cnf_matrix[0,0])/(cnf_matrix[1,1] + cnf_matrix[1,0]+ cnf_matrix[0,0] + cnf_matrix[0,1])
precision = cnf_matrix[1,1] / (cnf_matrix[1,1] + cnf_matrix[0,1]) 
F1 = 2 * precision * recall / (precision + recall) 
print('SVM test Recall : ', recall)
print('SVM test acc : ', acc)
print('SVM test precision ', precision)
print('SVM test F1 ', F1)
scores = cross_val_score(clf, X_train, Y_train.values.ravel(), cv=5)
print("SVM the mean of score:", scores.mean())
