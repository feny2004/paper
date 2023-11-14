# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 17:52:32 2021
for test only

@author: Lenovo
"""
import joblib
# from keras.models import Model
from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.utils import shuffle
import numpy as np 

x=np.load('test_x.npy',allow_pickle=True)
y=np.load('test_y.npy',allow_pickle=True)     
#标签
encoder = OneHotEncoder()
labels = encoder.fit_transform(y.reshape(-1,1)).toarray()
# labels= np_utils.to_categorical(Y_encoded)

scaler=MinMaxScaler()
x=scaler.fit_transform(x)

x,labels = shuffle(x,labels, random_state=1337) 

encoder=load_model('encoder.h5')

encoded_dims = encoder.predict(x)


clf = joblib.load('clf_3.pkl')

# score_train=clf.score(encoded_dims,labels)#如果改成三类，就要注释掉这行，在这里打分。

predict=clf.predict(encoded_dims)