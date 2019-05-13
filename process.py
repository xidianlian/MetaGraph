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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
import xgboost as xgb
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,classification_report,confusion_matrix


api_csv_path = "E:\\Spyder\\android_malware_detection\\input\\app_api_4000.csv"
perm_csv_path = "E:\\Spyder\\android_malware_detection\\input\\app_perm.csv"
method_csv_path = "E:\\Spyder\\android_malware_detection\\input\\app_method_15000.csv"
app_api_data = pd.read_csv(api_csv_path, dtype=np.uint16)
app_perm_data = pd.read_csv(perm_csv_path, dtype=np.uint16)
app_method_data = pd.read_csv(method_csv_path, dtype=np.int8)

label = pd.DataFrame(app_api_data['class'],columns=['class'])
app_api_data = app_api_data.drop(['class'], axis = 1)
app_perm_data = app_perm_data.drop(['class'], axis = 1)
app_method_data = app_method_data.drop(['class'], axis = 1)

new_api_data = app_api_data
new_method_data = app_method_data

#selector = SelectKBest(score_func=chi2, k=400)
#new_api_data = selector.fit_transform(app_api_data, label.values.ravel())
#selector1 = SelectKBest(score_func=chi2, k=800)
#new_method_data = selector1.fit_transform(app_method_data, label.values.ravel())
##合并
new_api_data = pd.DataFrame(new_api_data)
new_method_data = pd.DataFrame(new_method_data)

# 修改列名
col1 = []
col2 = []
for i in range(len(new_api_data.columns)):
    col1.append('I' + str(i))
for i in range(len(new_method_data.columns)):
    col2.append('M' + str(i))
new_api_data.columns = col1
new_method_data.columns = col2

new_data =  pd.concat([new_method_data], axis = 1) #new_api_data,app_perm_data,new_method_data

def normal_fun(new_data, label):
    
    # scoring = accuracy/f1/recall/precision
    score = ['precision', 'recall', 'f1','accuracy']
    # 这里只是为了打乱数据顺序
    # 用整个数据做交叉验证
    new_data, x_test, label, y_test = train_test_split(new_data, label, test_size=0.0,shuffle=True)
   
    #根据所有的交叉验证，计算相应的score 
    lr = LogisticRegression(C = 1, penalty='l1', solver="liblinear")
    scores = cross_validate(lr, new_data, label.values.ravel(), cv=5, scoring=score, return_train_score=False)
    print("LogisticRegression the mean of accuracy" + ":", scores['test_accuracy'].mean())
    print("LogisticRegression the mean of precision" + ":", scores['test_precision'].mean())
    print("LogisticRegression the mean of recall" + ":", scores['test_recall'].mean())
    print("LogisticRegression the mean of F1" + ":", scores['test_f1'].mean())
    print("-----------------------------")
    #
    rfc = RandomForestClassifier(random_state=1, n_estimators=200, min_samples_split=5,min_samples_leaf=3)
    scores = cross_validate(rfc, new_data, label.values.ravel(), cv=5, scoring=score)
    print("RandomForestClassifier the mean of accuracy" + ":", scores['test_accuracy'].mean())
    print("RandomForestClassifier the mean of precision" + ":", scores['test_precision'].mean())
    print("RandomForestClassifier the mean of recall" + ":", scores['test_recall'].mean())
    print("RandomForestClassifier the mean of F1" + ":", scores['test_f1'].mean())
    print("-----------------------------")
    
#    clf = svm.SVC(C=1, kernel='rbf', gamma='auto')
#    scores = cross_validate(clf, new_data, label.values.ravel(), cv=5,scoring=score)
#    print("SVM the mean of accuracy" + ":", scores['test_accuracy'].mean())
#    print("SVM the mean of precision" + ":", scores['test_precision'].mean())
#    print("SVM the mean of recall" + ":", scores['test_recall'].mean())
#    print("SVM the mean of F1" + ":", scores['test_f1'].mean())
#    print("-----------------------------")

def xgboost_fun(new_data, label):
    X_train, X_test, Y_train, Y_test = train_test_split(new_data, label, test_size=0.3,shuffle=True)
    
    print(Y_train['class'].value_counts())
    print(Y_test['class'].value_counts())
    lr = LogisticRegression(C = 1, penalty='l1', solver="liblinear")
    lr.fit(X_train, Y_train.values.ravel())
    Y_pre = lr.predict(X_test)
    cnf_matrix = confusion_matrix(Y_test, Y_pre)
    recall = cnf_matrix[1,1] / (cnf_matrix[1,0] + cnf_matrix[1,1]) 
    accuracy = (cnf_matrix[1,1] + cnf_matrix[0,0])/(cnf_matrix[1,1] + cnf_matrix[1,0]+ cnf_matrix[0,0] + cnf_matrix[0,1])
    precision = cnf_matrix[1,1] / (cnf_matrix[0,1] + cnf_matrix[1,1])
    F1 = 2 * precision * recall / (precision + recall) 
    print(cnf_matrix)
    print("LogisticRegression test accuracy :", accuracy)
    print("LogisticRegression test precision :", precision)
    print("LogisticRegression test recall :", recall)
    print("LogisticRegression test F1 :", F1)
    
    
    '''
    1、objective参数：[默认：reg:linear]
    定义需要被最小化的损失函数
    binary:logistic 二分类的逻辑回归，返回预测的概率(不是类别）
    multi:softmax 使用softmax的多分类器，返回预测的类别(不是概率)。[需要设num_class(类别数目)]
    multi:softprob 和multi:softmax参数一样，但是返回的是每个数据属于各个类别的概率。
    
    2、eval_metric[默认值取决于objective参数的取值]
    对于有效数据的度量方法。
    对于回归问题，默认值是rmse，对于分类问题，默认值是error。
    rmse 均方根误差；
    error 二分类错误率(阈值为0.5) merror 多分类错误率；
    logloss损失函数 mlogloss 多分类损失函数
    auc 曲线下面积
    
    3、early_stopping_rounds
    早期停止次数 ，假设为100，验证集的误差迭代到一定程度在100次内不能再继续降低，就停止迭代。
    这要求evals 里至少有 一个元素，如果有多个，按最后一个去执行。
    返回的是最后的迭代次数（不是最好的）。
    如果early_stopping_rounds 存在，则模型会生成三个属性，bst.best_score,bst.best_iteration,和bst.best_ntree_limit
    
    '''
    xgb_params = {
        'learning_rate': 0.1,  # 步长
        'n_estimators': 100,
        'max_depth': 6,  # 树的最大深度
        'objective': 'multi:softmax',
        'num_class': 2,
        'min_child_weight': 1,  # 决定最小叶子节点样本权重和，如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。
        'gamma': 0.3,  # 指定了节点分裂所需的最小损失函数下降值。这个参数的值越大，算法越保守
        'silent': 0,  # 输出运行信息
        'subsample': 0.8,  # 每个决策树所用的子样本占总样本的比例（作用于样本）
        'colsample_bytree': 0.8,  # 建立树时对特征随机采样的比例（作用于特征）典型值：0.5-1
        'nthread': 4,
        'seed': 27
    }
    xg_train = xgb.DMatrix(X_train, Y_train.values.ravel())
    xg_test = xgb.DMatrix(X_test, Y_test.values.ravel())
    watchlist = [(xg_train, 'train')]
    ## 先做交叉验证，调出较好参数
    #cv_res = xgb.cv(xgb_params,xg_train,num_boost_round=500,nfold=5,early_stopping_rounds=30,show_stdv=True)
    #print(cv_res)

    model = xgb.train(xgb_params, xg_train, 200, evals=watchlist)
    y_pre = model.predict(xg_test)

    print("xgboost   of accuracy" + ":", accuracy_score(Y_test,y_pre))
    print("xgboost   of precision(binary)" + ":", precision_score(Y_test,y_pre))
    print("xgboost   of recall(binary)" + ":", recall_score(Y_test, y_pre))
    print("xgboost   of F1(binary)" + ":", f1_score(Y_test,y_pre, average='binary'))
    cnf_matrix = confusion_matrix(Y_test, y_pre)
    print(cnf_matrix)
    
    
if __name__ == '__main__':
    normal_fun(new_data, label)
#    xgboost_fun(new_data, label)
