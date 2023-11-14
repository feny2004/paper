# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 11:51:44 2021

@author: Lenovo
"""


import numpy as np

import pandas as pd
import os,glob

home_dir = os.getcwd()
path = r'E:\leo数据\testdata'
# file  = glob.glob(os.path.join(path,"*.xlsx"))#仿真数据格式
file  = glob.glob(os.path.join(path,"*.csv"))
print(file)

data=[]

for f in file:
    data.append(pd.read_csv(f,dtype=str,header=None))
    datal=[]
    for x in range(len(data)):
        ts=data[x]
        ts.columns =['半长轴','标签']
        
        # tsl=ts.iloc[:,[1,2]]
        
        tsl=ts.values
        ts2=tsl.T
        ts_row=ts2[0,:].reshape(-1,1)
        # ts_row = ts_row.values
        b=ts2[1,1]
        ts2 = np.row_stack((ts_row,b))
        ts2=ts2.T
        datal.append(ts2)

  
dataset = np.array(datal)

dataset =np.squeeze(dataset,axis=1)

x = dataset[:,0:10]
y = dataset[:,-1]

np.save('test_x.npy',x)
np.save('test_y.npy',y)         
