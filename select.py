# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:57:38 2019

@author: lianWeiC
分别集成
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

INPUT_PATH = "E:\\Spyder\\android_malware_detection\\input"
OUTPUT_PATH = "E:\\Spyder\\android_malware_detection\\output"

api_csv_path = INPUT_PATH + "\\api_top_8000_total.csv"
perm_csv_path = INPUT_PATH + "\\perm.csv"
method_csv_path = INPUT_PATH + "\\methods_total_top_15000.csv"

api_csv_outpath = OUTPUT_PATH + "\\select_api_400.csv"
method_csv_outpath = OUTPUT_PATH + "\\select_method_400.csv"
perm_csv_outpath = OUTPUT_PATH + "\\app_perm_304.csv"
label_csv_outpath = OUTPUT_PATH + "\\label.csv"


def select_api():
    app_api_data = pd.read_csv(api_csv_path, dtype=np.int8)
    label = pd.DataFrame(app_api_data['class'], columns=["class"])
    label.to_csv(label_csv_outpath, index=False)
    app_api_data = app_api_data.drop(['class'], axis = 1)
    feature = app_api_data.columns
    selector = SelectKBest(score_func=chi2, k=400)
    new_api_data = selector.fit_transform(app_api_data, label.values.ravel())
    is_use = selector.get_support() #返回为true or false
    select_api_id = []
    for i in range(len(feature)):
        if is_use[i] == True:
            select_api_id.append(feature[i])
    new_api_data = pd.DataFrame(new_api_data, columns=select_api_id)
    select_api_id = pd.DataFrame(columns=select_api_id)
    # 输出csv
    select_api_id.to_csv(api_csv_outpath, index=False)
    return new_api_data

def select_method():
    app_method_data = pd.read_csv(method_csv_path, dtype=np.int8)
    label = pd.DataFrame(app_method_data['class'], columns=["class"])
    app_method_data = app_method_data.drop(['class'], axis = 1)
    feature = app_method_data.columns
    selector = SelectKBest(score_func=chi2, k=400)
    new_method_data = selector.fit_transform(app_method_data, label.values.ravel())
    is_use = selector.get_support()
    select_method_id = []
    for i in range(len(feature)):
        if is_use[i] == True:
            select_method_id.append(feature[i])
    new_method_data = pd.DataFrame(new_method_data, columns=select_method_id)
    select_method_id = pd.DataFrame(columns=select_method_id)
    select_method_id.to_csv(method_csv_outpath, index=False)
    return new_method_data

def select_perm():
    app_perm_data = pd.read_csv(perm_csv_path, dtype=np.int8)
    app_perm_data = app_perm_data.drop(['class'], axis = 1)
    app_perm_data.to_csv(perm_csv_outpath, index=False)
    return app_perm_data
    
if __name__ == "__main__":
  
    new_api_data = select_api()
    new_api_data.to_csv(OUTPUT_PATH + "\\app_api_400.csv", index=False)
    
    new_method_data = select_method()
    new_method_data.to_csv(OUTPUT_PATH + "\\app_method_400.csv",index=False)
    
    new_perm_data = select_perm()
    