# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 15:37:05 2019

@author: lianWeiC
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  


algorithm = ['DT','KNN','GBDT','LR','RF','XGB','FeatDroid']
api1 = [96.57, 96.32, 95.34, 96.78, 98.69, 98.51, 98.87]
perm1 = [95.86,95.95,95.53,94.99,96.57,96.45,96.75]
code1 = [97.67, 97.48, 96.34, 97.59, 97.63, 97.66, 97.74] 

api2 = [97.80,97.80,97.12, 97.88, 98.26, 98.32, 98.50]
perm2 = [96.95, 96.98, 96.42, 96.62, 97.49, 97.19, 97.55]
code2 = [97.88, 97.84, 96.32, 97.98, 98.08, 97.66, 98.11] 

api = api1
perm = perm1
code = code1


fig = plt.figure(figsize = (20,4))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)


ax1.bar(np.arange(7), api, 0.4, color = 'red')
ax1.set_xticklabels(algorithm,rotation=10)
tick_position = range(8)
ax1.set_xticks(tick_position)
ax1.set_ylim(90,100)
ax1.set_ylabel('F1(%)',fontsize=15)
ax1.set_title('API calls',fontsize=15)
ax1.tick_params(labelsize=13)
for a,b in zip(np.arange(7),api):
    ax1.text(a, b+0.001, '%.2f' % b, ha='center', va= 'bottom',fontsize=13)

ax2.bar(np.arange(7), perm, 0.4, color = 'red')
ax2.set_xticklabels(algorithm,rotation=10)
tick_position = range(8)
ax2.set_xticks(tick_position)
ax2.set_ylim(90,100)
#ax2.set_ylabel('F1',fontsize=15)
ax2.set_title('Permissions',fontsize=15)
ax2.tick_params(labelsize=13)
for a,b in zip(np.arange(7),perm):
    ax2.text(a, b+0.001, '%.2f' % b, ha='center', va= 'bottom',fontsize=13)

ax3.bar(np.arange(7), code, 0.4, color = 'red')
ax3.set_xticklabels(algorithm,rotation=10)
tick_position = range(8)
ax3.set_xticks(tick_position)
ax3.set_ylim(90,100)
#ax3.set_ylabel('F1',fontsize=15)
ax3.set_title('Code Blocks',fontsize=15)
ax3.tick_params(labelsize=13)
for a,b in zip(np.arange(7),code):
    ax3.text(a, b+0.001, '%.2f' % b, ha='center', va= 'bottom',fontsize=13)

ax1.grid(axis="y")
ax2.grid(axis="y")
ax3.grid(axis="y")

plt.savefig("C:\\Users\\72761\\Desktop\\实验\\论文插图\\AndroTracker.jpg",dpi=360)

plt.show()

 