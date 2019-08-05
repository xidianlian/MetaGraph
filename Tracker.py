# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:42:31 2019

@author: lianWeiC
"""
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt 
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.model_selection import cross_validate
import xgboost as xgb
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,classification_report,confusion_matrix

def chi_2(data,num,data_type):
    label = pd.DataFrame(data['class']).values.ravel()
    data = data.drop(['class'], axis = 1)
    col = data.columns 
    selector = SelectKBest(score_func=chi2, k=num)
    data = selector.fit_transform(data, label)
    data =  pd.DataFrame(data)
    index = selector.get_support(indices=True)
    score = selector.scores_
    p_value = selector.pvalues_
    
    support_data = {"index":col[index], "score":score[index], "p_value":p_value[index]}
    support_index = pd.DataFrame(support_data)
    support_index.to_csv("output\\support_index.csv", index=False, header=False)
    #修改列名
#    col = []
#    for i in range(len(data.columns)):
#        col.append(data_type+ str(i))
#    data.columns = col
    return data, label

def print_res(Y_test, y_pre, classter):
#    cnf_matrix = confusion_matrix(Y_test, y_pre)
#    print("TP : ", cnf_matrix[1,1])
#    print("FP : ", cnf_matrix[0,1])
#    print("TN : ", cnf_matrix[0,0])
#    print("FN : ", cnf_matrix[1,0])
    acc = accuracy_score(Y_test,y_pre)
    pre = precision_score(Y_test,y_pre)
    rec = recall_score(Y_test, y_pre)
    f1 = f1_score(Y_test,y_pre)
    print("%s accuracy:%.4f   precision:%.4f   recall:%.4f   F1:%.4f" % (classter, acc,pre,rec,f1))   
#    print(classter + " precision weight" + ":", precision_score(Y_test,y_pre,average='weighted'))
#    print(classter + " recall wight" + ":", recall_score(Y_test, y_pre,average='weighted'))
#    print(classter + " F1 wight" + ":", f1_score(Y_test,y_pre,average='weighted'))
#    print("***********************************")
   
      
def select_model(model_name):
    if model_name == "SVM":
        model = svm.SVC(kernel='rbf', C=1, gamma='auto')
    elif model_name == "GBDT":
        model = GradientBoostingClassifier(n_estimators = 50)
    elif model_name == "XGB":
        model = XGBClassifier(learning_rate= 0.1,n_estimators=100,max_depth=6, objective= 'multi:softmax',num_class=2,seed=27)
    elif model_name == "RF":
        model = RandomForestClassifier(random_state=1, n_estimators=20)
    elif model_name == "KNN":
        model = KNeighborsClassifier()
    elif model_name == "NB":
        model = MultinomialNB(alpha=2, class_prior=None, fit_prior=True)
    elif model_name == "DT":
        model = DecisionTreeClassifier(splitter='random')
    elif model_name == 'LR':
        model = LogisticRegression(C = 1, penalty='l1', solver="liblinear") 
    elif model_name == 'MLP':
        model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    else:
        pass
    return model 

def train_predict(modelname,X_train, X_test, Y_train, Y_test):
    clf = select_model(modelname)
    clf = clf.fit(X_train, Y_train)
    y_pre = clf.predict(X_test)
    print_res(Y_test, y_pre, modelname)

#对每一个分类器单独训练
def train_predict1(modelname,X_train, X_test, Y_train, Y_test):
    clf = select_model(modelname)
    for col in X_train.columns:
        clf = clf.fit(pd.DataFrame(X_train[col]), Y_train)
        y_pre = clf.predict(pd.DataFrame(X_test[col]))
        print_res(Y_test, y_pre, col)
    
    
def get_oof(clf,n_folds,X_train,y_train,X_test):
    ntrain = X_train.shape[0]
    ntest =  X_test.shape[0]
    kf = KFold(n_splits=n_folds, shuffle=True,random_state=2019)
    oof_train = np.zeros((ntrain,)) # 训练样本数 * 1
    oof_test = np.zeros((ntest,)) # 测试样本数 * 1
    
    for train_index, vald_index in kf.split(X_train):
        kf_X_train = X_train.iloc[train_index,:] # 数据
        kf_y_train = y_train[train_index] # 标签
        kf_X_vald = X_train.iloc[vald_index,:]  # k-fold的验证集
 
        clf.fit(kf_X_train, kf_y_train)
        oof_train[vald_index] = clf.predict(kf_X_vald)
 
        oof_test += clf.predict(X_test)
    oof_test = oof_test/float(n_folds)
    return oof_train, oof_test

def stacking(X_train, X_test, Y_train, Y_test,data_type):
    new_X_train = pd.DataFrame()
    new_X_test = pd.DataFrame()
    #['DT','KNN','NB','GBDT','LR','SVM','RF','MLP','XGB']
   
    models = ['DT','KNN','GBDT','LR','RF','XGB']
#    print("不集成的效果：only use %s"%(data_type))
#    for model in models:
#        train_predict(model,X_train, X_test, Y_train, Y_test)
    
    for model_name in models[0 : len(models)]:
        clf = select_model(model_name)
        model1_train, model1_test = get_oof(clf, 5, X_train, Y_train, X_test)
        new_X_train[model_name] = model1_train
        new_X_test[model_name] = model1_test
   
    return new_X_train,new_X_test


def FeatDroid(app_api_data,app_perm_data,app_method_data,label):
    X_api_train, X_api_test, Y_api_train, Y_api_test = train_test_split(app_api_data, label, test_size=0.3 ,random_state=2019,shuffle=True)
    X_perm_train, X_perm_test, Y_perm_train, Y_perm_test  = train_test_split(app_perm_data, label, test_size=0.3 ,random_state=2019,shuffle=True)
    X_method_train, X_method_test, Y_method_train, Y_method_test = train_test_split(app_method_data, label, test_size=0.3 ,random_state=2019,shuffle=True)
    
    '''
    new_X_api_train, new_X_api_test = stacking(X_api_train, X_api_test, Y_api_train, Y_api_test,"api")
    new_X_perm_train, new_X_perm_test = stacking(X_perm_train, X_perm_test, Y_perm_train, Y_perm_test,"perm")
    new_X_method_train, new_X_method_test = stacking(X_method_train, X_method_test, Y_method_train, Y_method_test,"method")

#    alpha = np.arange(0, 1.1, 0.1)
    alpha = np.arange(0.1, 1.0, 0.1)
    for i in alpha:
        for j in alpha:
            for k in alpha:
                if math.fabs(i + j + k - 1) < 0.0001:
                    print("i: %.1f, j: %.1f, k:%0.1f "%(i,j,k))
                    new_X_train = i*new_X_api_train  + j*new_X_perm_train  + k*new_X_method_train
                    new_X_test = i*new_X_api_test + j*new_X_perm_test  + k*new_X_method_test
                    print("融合训练：")
                    train_predict('RF',new_X_train,new_X_test,Y_perm_train, Y_perm_test)
    '''
    # 其他分类器
    models = ['DT','KNN','GBDT','LR','RF','XGB']
   
    new_X_train = pd.concat([X_api_train  + X_perm_train  + X_method_train],axis=1)
    new_X_test = pd.concat([X_api_test + X_perm_test  + X_method_test],axis=1)
    for model_name in models[0 : len(models)]:
        train_predict(model_name,new_X_train,new_X_test,Y_perm_train, Y_perm_test)
    
def cross_validation(clf, data, label):
    k = 5
    acc = 0
    pre = 0
    rec = 0
    f1 = 0
    kf = KFold(n_splits=k, shuffle=True)
    for train_index, vald_index in kf.split(data):
        kf_X_train = data.iloc[train_index,:]
        kf_X_vald = data.iloc[vald_index, :]
        kf_y_train = label[train_index]
        kf_y_test = label[vald_index]
        clf = clf.fit(kf_X_train, kf_y_train)
        kf_y_pre = clf.predict(kf_X_vald)
        acc += accuracy_score(kf_y_test,kf_y_pre)
        pre += precision_score(kf_y_test,kf_y_pre)
        rec += recall_score(kf_y_test,kf_y_pre)
        f1 += f1_score(kf_y_test,kf_y_pre)
    return acc/k, pre/k, rec/k, f1/k

def run_cross_vald(data, label):
    models = ['DT','KNN','GBDT','LR','RF','XGB']
    for model_name in models[0 : len(models)]:
        clf = select_model(model_name)
        acc,pre,rec,f1 = cross_validation(clf, data, label)
        print("cross_validation %s, acc %.4f, pre %.4f, rec %.4f, F1 %.4f" %(model_name,acc,pre,rec,f1))
     
# 单个分类器做交叉验证
def single_classfier_experiment(app_api_data,app_perm_data,app_method_data,label):
    run_cross_vald(app_api_data, label)
    run_cross_vald(app_perm_data, label)
    run_cross_vald(app_method_data, label)

def read_select_data(api_csv_path,perm_csv_path,method_csv_path):
    app_api_data = pd.read_csv(api_csv_path, dtype=np.uint16)
    app_perm_data = []
    app_method_data = []
    app_perm_data = pd.read_csv(perm_csv_path, dtype=np.uint16)
    app_method_data = pd.read_csv(method_csv_path, dtype=np.uint16)
    app_api_data, label = chi_2(app_api_data, 400,'api')
    app_perm_data, label = chi_2(app_perm_data, 'all', 'perm')
    app_method_data, label = chi_2(app_method_data, 1200,'method')
    return app_api_data,app_perm_data,app_method_data, label
  
def amd_data():
    print("-----------------------------------")
    print("-----------AMD data----------------")
    print("-----------------------------------")
    api_csv_path = "E:\\Spyder\\android_malware_detection\\input\\app_api_mal4000.csv"
    perm_csv_path = "E:\\Spyder\\android_malware_detection\\input\\app_perm.csv"
    method_csv_path = "E:\\Spyder\\android_malware_detection\\input\\app_method_mal5000.csv"
    app_api_data,app_perm_data,app_method_data, label = read_select_data(api_csv_path,perm_csv_path,method_csv_path)
#    single_classfier_experiment(app_api_data,app_perm_data,app_method_data,label)
    FeatDroid(app_api_data,app_perm_data,app_method_data,label)
    
def tracker_data():
    print("-----------------------------------")
    print("-----------Tracker data------------")
    print("-----------------------------------")
    api_csv_path = "E:\\Spyder\\android_malware_detection\\input\\Tracker_Api.csv"
    perm_csv_path = "E:\\Spyder\\android_malware_detection\\input\\Tracker_Perm.csv"
    method_csv_path = "E:\\Spyder\\android_malware_detection\\input\\Tracker_Method.csv"
    app_api_data,app_perm_data,app_method_data, label = read_select_data(api_csv_path,perm_csv_path,method_csv_path)
#    single_classfier_experiment(app_api_data,app_perm_data,app_method_data,label)
    FeatDroid(app_api_data,app_perm_data,app_method_data,label)

if __name__ == '__main__':

   # amd_data()

    tracker_data()
    
    
    