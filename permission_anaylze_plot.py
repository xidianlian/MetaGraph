# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:53:32 2019

@author: lianWeiC
"""
from decimal import Decimal
from sklearn import svm
from matplotlib.ticker import FuncFormatter
#from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt  
def permission():
    PERM_INPUT_PATH = "E:\\Spyder\\android_malware_detection\\input\\Tracker_Perm.csv"
    PERM_OUTPUT_PATH = "E:\\Spyder\\android_malware_detection\\output\\perm_support_index.csv"
    app_perm_data = pd.read_csv(PERM_INPUT_PATH, dtype=np.int8)
    label = pd.DataFrame(app_perm_data['class'], columns=["class"])
    app_perm_data = app_perm_data.drop(['class'], axis = 1)
    feature = app_perm_data.columns
    selector = SelectKBest(score_func=chi2, k=22)
    new_perm_data = selector.fit_transform(app_perm_data, label.values.ravel())
    scores = selector.scores_
    is_use = selector.get_support(indices = True) #返回为true or false
    print(scores[is_use])
    print(is_use)
    print(feature[is_use])
    
    data = { 'feature' : feature[is_use],
             'score' : scores[is_use]
           }
    select_perm_id = pd.DataFrame(data)
    select_perm_id = select_perm_id.sort_values(by="score",ascending=False)
    print(select_perm_id)
    select_perm_id = select_perm_id.drop('score',axis=1)
    # 输出csv
    select_perm_id.to_csv(PERM_OUTPUT_PATH, index=False)

def add_labels(rects):
    for rect in rects:
        width = rect.get_width()
        plt.text(width+0.5,rect.get_y() + rect.get_height()/2, '%.1f' %  width, ha='left', va='center')

def to_percent(temp, position):
    return '%1.0f' %(temp) + '%'
def fig():
    
#    plt.rc('font',family='Times New Roman',weight='normal')
    font_size = 8 # 字体大小
    fig_size = (10, 8) # 图表大小
    names = ('malware', 'benign') # 姓名
    subjects = ["READ_PHONE_STATE","SEND_SMS","RECEIVE_SMS","READ_EXTERNAL_STORAGE","READ_LOGS","READ_SMS",\
                "ACCESS_LOCATION_EXTRA_COMMANDS","ACCESS_COARSE_UPDATES","ACCESS_WIFI_STATE",\
                "ACCESS_COARSE_LOCATION","WRITE_EXTERNAL_STORAGE","MOUNT_UNMOUNT_FILESYSTEMS",\
                "KILL_BACKGROUND_PROCESSES","ACCESS_FINE_LOCATION","WAKE_LOCK",\
                "RECEIVE_BOOT_COMPLETED","RECORD_AUDIO","CALL_PHONE","READ_CONTACTS","VIBRATE"] # 科目
    malware = [1532,908,258,3755,4,48,70,41,1226,2364,2003,1302,145,37,2156,4527,1267,184,1028,3170]
    benign = [127,40,35,1620,64,9,3,0,290,812,89,52,58,22,541,3646,51,67,2,2603]
    
    subjects= subjects[::-1]
    malware = malware[::-1]
    benign = benign[::-1]
    
#    print(subjects[::-1])
    np.set_printoptions(formatter={'float': '{: 0.1f}'.format})
    scores = (np.array(malware) / 4554*100, 
              np.array(benign)/4551*100) # 成绩
  
    # 更新字体大小
    mpl.rcParams['font.size'] = font_size
    # 更新图表大小
    mpl.rcParams['figure.figsize'] = fig_size
    mpl.rcParams['font.weight'] = 'light'
    mpl.rcParams['font.family']='serif'
       # 设置柱形图宽度
    bar_width = 0.3
    
    index = np.arange(len(scores[0]))
    # 绘制「malware」的成绩
    rects1 = plt.barh(index, scores[0], bar_width, color='#c00000', label=names[0],edgecolor='#c00000')
    # 绘制「benign」的成绩
    rects2 = plt.barh(index + bar_width, scores[1], bar_width, color='#9BBB59', label=names[1])
    # X轴标题
    
    #plt.invert_yaxis()
    plt.yticks(index + bar_width/2, subjects)
    # Y轴范围
    plt.xlim(0,103.5)
    
    #plt.xlabel('percentage (%)',fontsize=8)
    # 图表标题
#    plt.title(u'企鹅班同学成绩对比')
    # 图例显示在图表下方
    plt.legend(loc='upper right')#, bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=5
    # 添加数据标签
    add_labels(rects1)
    add_labels(rects2)
    # 修改成百分比
    plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.tight_layout()
    plt.savefig("C:\\Users\\72761\\Desktop\\实验\\论文插图\\anaylze_perm.pdf",dpi=600)
    plt.show()
# 图表输出到本地
if __name__ == '__main__':
    fig()
  