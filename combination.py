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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.model_selection import cross_validate
import xgboost as xgb
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,classification_report,confusion_matrix
api_csv_path = "E:\\Spyder\\android_malware_detection\\input\\app_api_mal2000.csv"
#api_csv_path = "E:\\Spyder\\android_malware_detection\\input\\app_api_ben4000.csv"
#api_csv_path = "E:\\Spyder\\android_malware_detection\\input\\app_api_tot4000.csv"
perm_csv_path = "E:\\Spyder\\android_malware_detection\\input\\app_perm.csv"
method_csv_path = "E:\\Spyder\\android_malware_detection\\input\\app_method_mal5000.csv"
app_api_data = pd.read_csv(api_csv_path, dtype=np.uint16)
app_perm_data = pd.read_csv(perm_csv_path, dtype=np.uint16)
app_method_data = pd.read_csv(method_csv_path, dtype=np.uint16)

label = pd.DataFrame(app_perm_data['class']).values.ravel()
app_perm_data = app_perm_data.drop(['class'], axis=1)
app_api_data = app_api_data.drop(['class'],axis=1)
app_method_data = app_method_data.drop(['class'],axis=1)

selector = SelectKBest(score_func=chi2, k=400)
new_api_data = selector.fit_transform(app_api_data, label)
selector1 = SelectKBest(score_func=chi2, k=800)
new_method_data = selector1.fit_transform(app_method_data, label)
app_api_data = pd.DataFrame(new_api_data)
app_method_data = pd.DataFrame(new_method_data)
#修改列名
col1 = []
col2 = []
for i in range(len(app_api_data.columns)):
    col1.append('API' + str(i))
for i in range(len(app_method_data.columns)):
    col2.append('METHOD' + str(i))
app_api_data.columns = col1
app_method_data.columns = col2

test_radio = 0.3
X_perm_train, X_perm_test, Y_perm_train, Y_perm_test = train_test_split(app_perm_data ,label, test_size=test_radio,random_state=2019,shuffle=True)
X_api_train, X_api_test, Y_api_train, Y_api_test = train_test_split(app_api_data ,label, test_size=test_radio ,random_state=2019,shuffle=True)
X_method_train, X_method_test, Y_method_train, Y_method_test = train_test_split(app_method_data ,label, test_size=test_radio ,random_state=2019,shuffle=True)


def print_res(Y_test, y_pre, classter):
    cnf_matrix = confusion_matrix(Y_test, y_pre)
    TP = cnf_matrix[1,1]
    FP = cnf_matrix[0,1]
    TN = cnf_matrix[0,0]
    FN = cnf_matrix[1,0]
    mal_num = TP + FN
    ben_num = TN + FP
    
    mal_recall = TP / (TP + FN)
    mal_precision = TP / (TP + FP)
    M_F1 = 2 * mal_recall * mal_precision / (mal_recall + mal_precision)
    
    ben_recall = TN / ( TN + FP)
    ben_precision = TN / ( TN + FN)
    B_F1 = 2 * ben_recall * ben_precision / (ben_precision + ben_recall)
    
    W_F1 = (ben_num * B_F1 + mal_num * M_F1) / (ben_num + mal_num)
    print("malware:", mal_num)
    print("benign:", ben_num)
    print("TP : ", TP)
    print("FP : ", FP)
    print("TN : ", TN)
    print("FN : ", FN)
    print(classter + " accuracy" + ":", accuracy_score(Y_test,y_pre))
    # TP/(TP+FP)
    print(classter + " M-precision" + ":", precision_score(Y_test,y_pre))
    # TP/(TP+FN)
    print(classter + " M-recall" + ":", recall_score(Y_test, y_pre))
    
    print(classter + " M-F1" + ":", f1_score(Y_test, y_pre))
    # TN/(TN+FN)
    print(classter + " B-precision" + ":", ben_precision)
    # TN/(TN+FP)
    print(classter + " B-recall" + ":", ben_recall)
    print(classter + " B-F1" + ":", B_F1)
    print(classter + "W-F1" + ":", f1_score(Y_test,y_pre,average='weighted'))
    print("***********************************")
      
def select_model(model_name):
    if model_name == "SVM":
        model = svm.SVC(kernel='rbf', C=1, gamma='auto')
    elif model_name == "GBDT":
        model = GradientBoostingClassifier(n_estimators=100)
    elif model_name == "XGB":
        model = XGBClassifier(learning_rate= 0.1,n_estimators=100,objective= 'multi:softmax',num_class=2,
                min_child_weight=1,max_depth=6,gamma=0.3,subsample=0.8,silent=0,seed=27)
    elif model_name == "RF":
        model = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=2,min_samples_leaf=2)
    elif model_name == "KNN":
        model = KNeighborsClassifier()
    elif model_name == "NB":
        model = MultinomialNB(alpha=2, class_prior=None, fit_prior=True)
    elif model_name == "DT":
        model = DecisionTreeClassifier()
    elif model_name == 'LR':
        model = LogisticRegression(C = 1, penalty='l1', solver="liblinear") 
#    elif model_name == 'MLP':
#        model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    else:
        pass
    return model 
def train_predict(modelname,X_train, X_test, Y_train, Y_test):
    clf = select_model(modelname)
    clf = clf.fit(X_train, Y_train)
    y_pre = clf.predict(X_test)
    print_res(Y_test, y_pre, modelname)
    

def get_oof(clf,n_folds,X_train,y_train,X_test):
    ntrain = X_train.shape[0]
    ntest =  X_test.shape[0]
    kf = KFold(n_splits=n_folds, shuffle=False)
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

def stacking(X_train, X_test, Y_train, Y_test):
    new_X_train = pd.DataFrame()
    new_X_test = pd.DataFrame()
    #['DT','KNN','NB','GDBC','LR','RF','SVM','MLP','XGB','LR']
    #['DT','LR','RF','LR']
    models = ['DT','KNN','GBDT','LR','RF']
    print("不集成的效果：")
    for model in models:
        train_predict(model,X_train, X_test, Y_train, Y_test)
    
    for model_name in models[0 : len(models) - 1]:
        clf = select_model(model_name)
        model1_train, model1_test = get_oof(clf, 5, X_train, Y_train, X_test)
        new_X_train[model_name] = model1_train
        new_X_test[model_name] = model1_test
    
    return new_X_train, new_X_test 
    

    
    
if __name__ == '__main__':
    new_X_perm_train, new_X_perm_test = stacking(X_perm_train, X_perm_test, Y_perm_train, Y_perm_test)
    #    # 第二次训练
    train_predict("LR",new_X_perm_train,new_X_perm_test,Y_perm_train,Y_perm_test)
    
