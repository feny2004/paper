# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 09:48:41 2021

@author: Lenovo
"""

import scipy.io as scio

import numpy as np
np.random.seed(1443)
from tensorflow.keras.models import Sequential
from keras.layers import SimpleRNN,Dense,LSTM

from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
data = scio.loadmat('E:\LEOnew\model_data.mat')
dataset=data['dd']
np.random.shuffle(dataset)

data_t=dataset[:,0:20]
labels=dataset[:,20]

encode=OneHotEncoder()
label=encode.fit_transform(labels.reshape(-1,1)).toarray()

x_train,x_test,y_train,y_test = train_test_split(data_t,label
                                                 ,test_size=0.2
                                                 ,random_state=1443)

scale = StandardScaler()
scaler=scale.fit(x_train)
mean_X, std_X = scaler.mean_, scaler.scale_
x_train=scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential()
model.add(LSTM(units = 80
                    ,batch_input_shape = (None,1,20)
                    ,unroll = True))

model.add(Dense(units=4,activation='softmax'))

model.summary()
adam = Adam(lr=0.001)
model.compile(optimizer=adam,
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])
x_train = np.expand_dims(x_train,axis=1)
model.fit(x_train, y_train, epochs=50, batch_size=128,validation_split=0.2)
score_train=model.evaluate(x_train,y_train)
x_test_t =np.expand_dims(x_test,axis=1)

test_preds =model.predict_classes(x_test_t)
scores = model.evaluate(x_test_t, y_test, verbose=0)

print("Saving model to disk \n")
mp = "E:\LEOnew\clf_model.h5"
model.save(mp)

add_test = scio.loadmat(r'E:\\LEOnew\\add_test.mat')
test=add_test['dd']
x=test[:,0:20]
X=scaler.transform(x)
X=np.expand_dims(X,axis=1) 
np.save(r'testX.npy',X)


