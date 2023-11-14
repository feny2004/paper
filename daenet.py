# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 21:07:21 2021
特征提取，降噪自编码器.
@author: ff
"""

from sklearn.model_selection import train_test_split
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
# import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
# import tensorflow.compat.v1 as tf

x=np.load('x.npy',allow_pickle=True)
y=np.load('y.npy',allow_pickle=True)     
#标签
encoder = OneHotEncoder()
labels = encoder.fit_transform(y.reshape(-1,1)).toarray()
# labels= np_utils.to_categorical(Y_encoded)

scaler=MinMaxScaler()
x=scaler.fit_transform(x)

# np.random.seed(42)
X_train,X_test,y_train,y_test = train_test_split(x,labels,test_size=0.3,random_state=2)  

np.savetxt(r'E:\leo数据\X_train.csv',X_train,delimiter=',',fmt ='%s')
np.savetxt(r'E:\leo数据\X_test.csv',X_test,delimiter=',',fmt ='%s')
np.savetxt(r'E:\leo数据\y_train.csv',y_train,delimiter=',',fmt ='%s')
np.savetxt(r'E:\leo数据\y_test.csv',y_test,delimiter=',',fmt ='%s')


encoding_dim = 3#超参数，可以改变，即希望特征压缩的维度

#输入

input_dim = Input(shape=(10,))

#单层降噪自编码层
'''
encoded = Dense(encoding_dim, activation='relu')(input_dim)

decoded = Dense(10, activation='sigmoid')(encoded)

autoencoder = Model(inputs=input_dim, outputs=decoded)

encoder = Model(inputs=input_dim, outputs=encoded)

encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]

decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='mse')
#
'''
#堆叠降噪自编码

encoded1 = Dense(8, activation='relu')(input_dim) 
encoded2 = Dense(5, activation='relu')(encoded1)
encoder_output = Dense(encoding_dim)(encoded2)

decoded1 = Dense(5, activation='relu')(encoder_output)
decoded2 = Dense(8, activation='relu')(decoded1)  
decoded3 = Dense(10, activation='sigmoid')(decoded2) 

#构建自编码模型
autoencoder = Model(inputs=input_dim, outputs=decoded3)

#编码模型
encoder = Model(inputs=input_dim, outputs=encoder_output)#这是截取从原模型的输入层到中间层的新模型
autoencoder.summary()#显示网络结构
encoder.summary()

autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(X_train, X_train, epochs=50, batch_size=3, 
                shuffle=True,validation_data=(X_test,X_test))

encoder.save('E:\\leo数据\\encoder.h5')
autoencoder.save('E:\\leo数据\\autoencoder.h5')

encoded_dims = encoder.predict(X_train)#新模型调用predict函数即可得到相当于原模型的中间层的输出结果
encoded_dims1 = encoder.predict(X_test)
decoded_dims = autoencoder.predict(X_train)
decoded_dims1 = autoencoder.predict(X_test)

np.savetxt(r'E:\leo数据\encoded_train.csv',encoded_dims,delimiter=',',fmt ='%s') 
np.savetxt(r'E:\leo数据\encoded_test.csv',encoded_dims1,delimiter=',',fmt ='%s') 
