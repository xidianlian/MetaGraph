# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:47:51 2019

@author: lianWeiC
"""
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

INPUT_PATH = "E:\\Spyder\\android_malware_detection\\output"
OUTPUT_PATH = "E:\\Spyder\\android_malware_detection\\PrecomputedKernels"

app_api_path = INPUT_PATH + "\\app_api_400.csv"
app_perm_path = INPUT_PATH + "\\app_perm_304.csv"
api_pack_path = INPUT_PATH + "\\api_pack.csv"
api_method_path = INPUT_PATH + "\\method_api.csv"
label_path = INPUT_PATH + "\\label.csv"

train_kernel_path = OUTPUT_PATH + "\\meta4"
test_kernel_path = OUTPUT_PATH + "\\meta4.test"

def get_data():
    app_api = pd.read_csv(app_api_path ,dtype=np.uint16) # shape(20807, 400)
    app_perm = pd.read_csv(app_perm_path ,dtype=np.uint16)# shape(20807, 304)
    
    data = pd.concat([app_api, app_perm], axis=1)
    data = data.as_matrix() # shape(20807, 704)
    
    label = pd.read_csv(label_path, dtype=np.int8)
    label = label.values.ravel()
    return train_test_split(data, label, test_size=0.2, shuffle=False)

def output_file(data, path):
    """ 输出核矩阵"""
    pd.DataFrame(data).to_csv(path,index=False)
#    shape = np.array([np.array(data.shape)])
#    np.savetxt(path, shape, fmt='%d')
#    app_data = pd.DataFrame(data)
#    app_data.to_csv(path,index=False,header=False,mode="a", sep='\t')
    #np.savetxt(path,data,fmt="%d")

def get_train_kernel(train_data, api_method, api_pack):
    app_api_mat = train_data[:, 0:400]

    api_method_api = np.dot(api_method, api_method.T)
    api_pack_api = np.dot(api_pack, api_pack.T)
    api_api = api_method_api * api_pack_api
    
    app_api_api = np.dot(app_api_mat, api_api)
    app_api_api_app = np.dot(app_api_api, app_api_mat.T)
    
    output_file(app_api_api_app, train_kernel_path)
    return app_api_api_app # shape(16645, 16645)

def get_test_kernel(train_data, test_data, api_method, api_pack):
    train_app_api_mat = train_data[:, 0:400]
    
    test_app_api_mat = test_data[:, 0:400]
    
    api_method_api = np.dot(api_method, api_method.T)
    api_pack_api = np.dot(api_pack, api_pack.T)
    api_api = api_method_api * api_pack_api
    
    app_api_api = np.dot(test_app_api_mat, api_api)
    app_api_api_app = np.dot(app_api_api, train_app_api_mat.T)
    
    output_file(app_api_api_app, test_kernel_path)
    return app_api_api_app
    
def svm_test(train_kernel, test_kernel, Y_train, Y_test):
    clf = svm.SVC(C=2, kernel='precomputed', gamma='auto')
    clf.fit(train_kernel, Y_train)
    Y_pre = clf.predict(test_kernel)
    cnf_matrix = confusion_matrix(Y_test, Y_pre)
    recall = cnf_matrix[1,1] / (cnf_matrix[1,1] + cnf_matrix[1,0]) 
    acc = (cnf_matrix[1,1] + cnf_matrix[0,0])/(cnf_matrix[1,1] + cnf_matrix[1,0]+ cnf_matrix[0,0] + cnf_matrix[0,1])
    precision = cnf_matrix[1,1] / (cnf_matrix[1,1] + cnf_matrix[0,1]) 
    F1 = 2 * precision * recall / (precision + recall) 
    print('SVM test Recall : ', recall)
    print('SVM test acc : ', acc)
    print('SVM test precision ', precision)
    print('SVM test F1 ', F1)
#    scores = cross_val_score(clf, X_train, Y_train.values.ravel(), cv=5)
#    print("SVM the mean of score:", scores.mean())

if __name__ == "__main__":
    
    api_pack = pd.read_csv(api_pack_path, dtype=np.int32) # shape(400, 301)
    api_pack = api_pack.as_matrix() 
    
    method_api = pd.read_csv(api_method_path, dtype=np.int32) # shape(400, 400)
    method_api = method_api.as_matrix()
    api_method = method_api.T;
    
    X_train, X_test, Y_train, Y_test =get_data()
    train_kernel = get_train_kernel(X_train, api_method, api_pack)
    test_kernel = get_test_kernel(X_train, X_test, api_method, api_pack)
    
    #svm_test(train_kernel, test_kernel , Y_train, Y_test)

