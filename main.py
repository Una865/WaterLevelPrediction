import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
from tensorflow import keras
from sklearn.model_selection import train_test_split

import sys
import ctypes
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def load_data(filename):
    df = pd.read_excel(filename)
    data = df.to_numpy()
    np.random.shuffle(data)
    y = data[:,-1]
    x = data[:,:-1]

    return x,y

def load_test_data(filename):
    df = pd.read_excel(filename)
    data = df.to_numpy()
    return data

hidden_units1 = 8
hidden_units2 = 32
learning_rate = 0.0001

x,y = load_data('KSC.xlsx')
x_train,x_test,y_train, y_test = train_test_split(x, y, test_size = 0.2)

tf.random.set_seed(42)
model = tf.keras.Sequential([tf.keras.layers.Dense(hidden_units1,activation = 'relu'),
                              tf.keras.layers.Dense(hidden_units2,activation = 'relu'),
                              tf.keras.layers.Dense(1,  activation='linear')])

msle = tf.keras.losses.MeanSquaredLogarithmicError()
opt = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(
    loss = msle,
    optimizer = opt,
    metrics = ['msle'])

model.fit(x_train,y_train, epochs = 30,batch_size = 64,validation_split = 0.2)
predy = model.predict(x_test)
loss = 0
n,m = predy.shape
for i in range(m):
    loss+=(y_test[i]-predy[i])**2
print("Loss on the test data: {}".format(loss))
x_predict = load_test_data('KSC-dpred3.xlsx')
y_predict = model.predict(x_predict)

df = pd.DataFrame(y_predict.copy())
filepath = 'KSC-pred3.xlsx'
df.to_excel(filepath,index = False)