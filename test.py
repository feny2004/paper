# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 13:54:36 2021

@author: Lenovo
"""

from keras.models import load_model
import numpy as np
import scipy.io as scio

load_model = load_model("E:\LEOnew\clf_model.h5")
X =np.load('testX.npy',allow_pickle=True, encoding="latin1")
# add_test = scio.loadmat(r'E:\\LEOnew\\add_test.mat')
# test=add_test['dd']
# test_x=test[:,0:10]
# X=np.expand_dims(test_x,axis=1) 
test_predict =load_model.predict_classes(X)

add_test=np.reshape(X,(1925,20))

add_testd=np.column_stack((add_test,test_predict))
scio.savemat('test.mat',{'dd':add_testd})
