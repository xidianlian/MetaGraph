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

def select_features(data,label,num):
    selector = SelectKBest(score_func=chi2, k=num)
    new_api_data = selector.fit_transform(data, label)
    return pd.DataFrame(new_api_data)

def split_data(data,k_features):
    label = pd.DataFrame(data['class'])
    data = data.drop(['class'], axis = 1)
    data = select_features(data, label.values.ravel(), k_features)
    data['class'] = label
    mal_data = data[data['class'].isin([1])]
    mal_data = mal_data.drop(['class'], axis = 1)
    ben_data = data[data['class'].isin([0])]
    ben_data = ben_data.drop(['class'],axis=1)
    return mal_data,ben_data

def under_sampling(mal_data, ben_data):
    mal_num = len(mal_data)
    ben_num = len(ben_data)
    test_size_ben_radio = 0.3
    test_size_mal_radio = ben_num * test_size_ben_radio / mal_num 
     
    # 得到恶意软件和良性软件各自的训练集和测试集
    mal_data_train, mal_data_test ,mal_y_train,mal_y_test = train_test_split(mal_data, np.ones(len(mal_data), dtype=np.uint16), test_size=test_size_mal_radio ,random_state=2019,shuffle=True)
    ben_data_train, ben_data_test ,ben_y_train,ben_y_test = train_test_split(ben_data, np.zeros(len(ben_data), dtype=np.uint16), test_size=test_size_ben_radio ,random_state=2019,shuffle=True)
    
    X_test = pd.concat([mal_data_test, ben_data_test], axis = 0)
    Y_test = np.append(mal_y_test, ben_y_test)
    
    # --------------start下采样--------------------
    mal_select_radio = len(ben_data_train) / len(mal_data_train) 
    x1, mal_use_data, y1, y2 = train_test_split(mal_data_train, mal_y_train, test_size=mal_select_radio,random_state=2019,shuffle=True)
    # --------------end下采样--------------------
    X_train = pd.concat([mal_use_data, ben_data_train], axis = 0)
    Y_train = np.append(y2, ben_y_train)
    return X_train,X_test,Y_train,Y_test

'''
# 按列分开
mal_perm_shape = mal_perm_data.shape
mal_api_shape = mal_api_data.shape
mal_method_shape = mal_method_data.shape
cut1 = mal_perm_shape[1]
cut2 = cut1 + mal_api_shape[1]
cut3 = cut2 + mal_method_shape[1]
mal_perm_data = mal_data.iloc[:, 0:cut1]
mal_api_data = mal_data.iloc[:, cut1:cut2]
mal_method_data = mal_data.iloc[:, cut2:cut3]

label1 = np.ones(len(mal_data), dtype=np.uint16)
label0 = np.zeros(len(ben_data), dtype=np.uint16)
label = np.append(label1, label0)


app_perm_data = pd.concat([mal_perm_data, ben_perm_data], axis = 0)
app_api_data = pd.concat([mal_api_data, ben_api_data], axis = 0)
app_method_data = pd.concat([mal_method_data, ben_method_data], axis = 0)

selector = SelectKBest(score_func=chi2, k=400)
new_api_data = selector.fit_transform(app_api_data, label)
selector1 = SelectKBest(score_func=chi2, k=800)
new_method_data = selector1.fit_transform(app_method_data, label)
selector2 = SelectKBest(score_func=chi2, k=361)
new_perm_data = selector2.fit_transform(app_perm_data, label)

new_api_data = pd.DataFrame(new_api_data)
new_perm_data = pd.DataFrame(new_perm_data)
new_method_data = pd.DataFrame(new_method_data)
'''
def print_res(Y_test, y_pre, classter):
    cnf_matrix = confusion_matrix(Y_test, y_pre)
    print("TP : ", cnf_matrix[1,1])
    print("FP : ", cnf_matrix[0,1])
    print("TN : ", cnf_matrix[0,0])
    print("FN : ", cnf_matrix[1,0])
    print(classter + " accuracy" + ":", accuracy_score(Y_test,y_pre))
    print(classter + " precision" + ":", precision_score(Y_test,y_pre))
    print(classter + " recall" + ":", recall_score(Y_test, y_pre))
    print(classter + " F1" + ":", f1_score(Y_test,y_pre))
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
    #['DT','KNN','NB','GBDT','LR','SVM','RF','MLP','XGB']
    models = ['DT','GBDT','LR','RF','MLP']
    print("不集成的效果：")
#    for model in models:
#        train_predict(model,X_train, X_test, Y_train, Y_test)
    
    for model_name in models[0 : len(models)]:
        clf = select_model(model_name)
        model1_train, model1_test = get_oof(clf, 5, X_train, Y_train, X_test)
        new_X_train[model_name] = model1_train
        new_X_test[model_name] = model1_test
   
    return new_X_train,new_X_test
  
if __name__ == '__main__':
    api_csv_path = "E:\\Spyder\\android_malware_detection\\input\\app_api_mal2000.csv"
    perm_csv_path = "E:\\Spyder\\android_malware_detection\\input\\app_perm.csv"
    method_csv_path = "E:\\Spyder\\android_malware_detection\\input\\app_method_mal5000.csv"
    app_api_data = pd.read_csv(api_csv_path, dtype=np.uint16)
    app_perm_data = pd.read_csv(perm_csv_path, dtype=np.uint16)
    app_method_data = pd.read_csv(method_csv_path, dtype=np.uint16)
    
    mal_api_data, ben_api_data = split_data(app_api_data, 400)
    mal_perm_data, ben_perm_data = split_data(app_perm_data,300)
    mal_method_data, ben_method_data = split_data(app_method_data,400)
    X_api_train, X_api_test, Y_api_train, Y_api_test = under_sampling(mal_api_data, ben_api_data)
    X_perm_train, X_perm_test, Y_perm_train, Y_perm_test = under_sampling(mal_perm_data, ben_perm_data)
    X_method_train, X_method_test, Y_method_train, Y_method_test = under_sampling(mal_method_data, ben_method_data)    


#   stacking 第一次训练    
    new_X_perm_train, new_X_perm_test = stacking(X_perm_train, X_perm_test, Y_perm_train, Y_perm_test)
    new_X_api_train, new_X_api_test = stacking(X_api_train, X_api_test, Y_api_train, Y_api_test)
    new_X_method_train, new_X_method_test = stacking(X_method_train, X_method_test, Y_method_train, Y_method_test)

#    print("单独训练: permission、api、method")
#    train_predict('XGB',new_X_perm_train,new_X_perm_test,Y_perm_train, Y_perm_test)
#    train_predict('XGB',new_X_api_train,new_X_api_test,Y_perm_train, Y_perm_test)
#    train_predict('XGB',new_X_method_train,new_X_method_test,Y_perm_train, Y_perm_test)
    # 方法一
#    new_X_train = (new_X_perm_train + new_X_api_train  + new_X_method_train) / 3
#    new_X_test = (new_X_perm_test + new_X_api_test  + new_X_method_test) / 3
    # 方法二
    new_X_train = pd.concat([new_X_perm_train, new_X_api_train, new_X_method_train], axis = 1)
    new_X_test = pd.concat([new_X_perm_test, new_X_api_test, new_X_method_test], axis = 1)
    # 方法三
#    alpha = np.arange(0.1, 1, 0.1)
#  
#    for i in alpha:
#        for j in alpha:
#            for k in alpha:
#                if i + j + k == 1:
#                    print("i: %.2f, j: %.2f, k:%0.2f "%(i,j,k))
#                    new_X_train = i*new_X_perm_train + j*new_X_api_train  + k*new_X_method_train
#                    new_X_test = (new_X_perm_test + new_X_api_test  + new_X_method_test)/3
#                    print("融合训练：")
#                    train_predict('LR',new_X_train,new_X_test,Y_perm_train, Y_perm_test)
                    
#    # 第二次训练
    print("融合训练：")
    train_predict('LR',new_X_train,new_X_test,Y_perm_train, Y_perm_test)
    
    print("组合训练：")
    X_train = pd.concat([X_api_train, X_perm_train, X_method_train], axis = 1)
    X_test = pd.concat([X_api_test, X_perm_test, X_method_test], axis = 1)
     #   stacking 第一次训练    
    new_X_train, new_X_test = stacking(X_train, X_test, Y_perm_train, Y_perm_test)
   
    # 第二次训练
    print("组合训练：")
    train_predict('LR',new_X_train,new_X_test,Y_perm_train, Y_perm_test)
    