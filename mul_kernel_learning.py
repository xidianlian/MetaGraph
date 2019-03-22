# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:48:01 2019

@author: lianWeiC
"""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

INPUT_PATH = "E:\\Spyder\\android_malware_detection\\PrecomputedKernels"

train_label_path = INPUT_PATH + "\\y_train"
test_label_path = INPUT_PATH + "\\y_test"

meta1_train_path = INPUT_PATH + "\\meta1"
meta1_test_path = INPUT_PATH + "\\meta1.test"
meta2_train_path = INPUT_PATH + "\\meta2"
meta2_test_path = INPUT_PATH + "\\meta2.test"
meta3_train_path = INPUT_PATH + "\\meta3"
meta3_test_path = INPUT_PATH + "\\meta3.test"
meta4_train_path = INPUT_PATH + "\\meta4"
meta4_test_path = INPUT_PATH + "\\meta4.test"
meta5_train_path = INPUT_PATH + "\\meta5"
meta5_test_path = INPUT_PATH + "\\meta5.test"

def getdata(train_data_path, test_data_path):
    train_data = pd.read_csv(train_data_path, dtype=np.uint32)
    test_data = pd.read_csv(test_data_path, dtype=np.uint32)
    train_data = train_data.as_matrix()
    test_data = test_data.as_matrix()
    print(train_data.shape)
    print(test_data.shape)
    return train_data, test_data
def svm_test(train_kernel, test_kernel, Y_train):
    clf = svm.SVC(C=2, kernel='precomputed', gamma='auto')
    clf.fit(train_kernel, Y_train)
    Y_pre = clf.predict(test_kernel)
    return Y_pre

if __name__ == "__main__":
   Y_train = np.loadtxt(train_label_path)
   Y_test = np.loadtxt(test_label_path)
   
   train_data, test_data = getdata(meta1_train_path, meta1_test_path)
   Y_pre1 = svm_test(train_data, test_data, Y_train)
   
   train_data, test_data = getdata(meta2_train_path, meta2_test_path)
   Y_pre2 = svm_test(train_data, test_data, Y_train)
   
   train_data, test_data = getdata(meta3_train_path, meta3_test_path)
   Y_pre3 = svm_test(train_data, test_data, Y_train)
   
   train_data, test_data = getdata(meta4_train_path, meta4_test_path)
   Y_pre4 = svm_test(train_data, test_data, Y_train)
   
   train_data, test_data = getdata(meta5_train_path, meta5_test_path)
   Y_pre5 = svm_test(train_data, test_data, Y_train)
   
   Y_pre = []
   for i in range(len(Y_pre1)):
       cnt = Y_pre1[i] + Y_pre2[i] + Y_pre3[i] + Y_pre4[i] + Y_pre5[i]
       Y_pre.append(1 if cnt > 0  else -1)
   cnf_matrix = confusion_matrix(Y_test, Y_pre)
   recall = cnf_matrix[1,1] / (cnf_matrix[1,1] + cnf_matrix[1,0]) 
   acc = (cnf_matrix[1,1] + cnf_matrix[0,0])/(cnf_matrix[1,1] + cnf_matrix[1,0]+ cnf_matrix[0,0] + cnf_matrix[0,1])
   precision = cnf_matrix[1,1] / (cnf_matrix[1,1] + cnf_matrix[0,1]) 
   F1 = 2 * precision * recall / (precision + recall) 
   print('SVM test Recall : ', recall)
   print('SVM test acc : ', acc)
   print('SVM test precision ', precision)
   print('SVM test F1 ', F1)
   #scores = cross_val_score(clf, X_train, Y_train.values.ravel(), cv=5)
   #print("SVM the mean of score:", scores.mean())
   
   