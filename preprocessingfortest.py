# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 15:31:26 2021

@author: Lenovo
"""

import pandas as pd 
import numpy as np 
import glob, os
from sklearn.preprocessing import MinMaxScaler

def _slide_window(rows, sw_width, sw_steps):
    '''
    函数功能：
    按指定窗口宽度和滑动步长实现单列数据截取,此处先对半长轴来进行操作
    --------------------------------------------------
    参数说明：
    rows：单个文件中的行数；
    sw_width：滑动窗口的窗口宽度；
    sw_steps：滑动窗口的滑动步长；
    '''
    start = 0
    s_num = (rows - sw_width) // sw_steps # 计算滑动次数
    new_rows = sw_width + (sw_steps * s_num) # 完整窗口包含的行数，丢弃少于窗口宽度的采样数据；
    
    while True:
        if (start + sw_width) > new_rows: # 如果窗口结束索引超出最大索引，结束截取；
            return
        yield start, start + sw_width
        start += sw_steps
        
import scipy.io as scio

home_dir =os.getcwd()

# path = r'E:\leo数据\sate_class_1'
path = r'E:\LEOnew\LEOnew\星链第一批数据集 - 无预处理\星链第一批数据集 - 无预处理\label1_down'
file = glob.glob(os.path.join(path, "*.mat"))
print(file)

data = []
for i,j in zip(file,range(len(file))):
    data = scio.loadmat(i)
    data1 = data.get('data')
    data2= pd.DataFrame(data1.T)
    # data2.columns=['编号','半长轴','时间']
    # cols = list(data2)
    # cols.insert(0,cols.pop(cols.index('时间')))
    # d=data2.loc[:,cols]#按照clos排列程新的矩阵
    d=data2.iloc[:,1]
    for start,end in _slide_window(len(d),10,5):
        data3 = d.values[start:end,]
        data3 = np.column_stack([pd.DataFrame(data3),np.ones(len(data3))*2])#这里记得，爬升换成zeros,碎片换成ones，此处2是卫星。
        np.savetxt(r'E:\leo数据\sate_class_1\1\sat{}{}.csv'.format(j,end),data3,delimiter=',',fmt ='%s')
        # dataset.append(data3)
    


