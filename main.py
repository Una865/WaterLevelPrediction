import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import openpyxl

def load_data(filename = 'Sava.xlsx'):
    df = pd.read_excel('KFC.xlsx')
    data = df.to_numpy()
    y = data[:,1]
    x = np.delete(data,1,1)

    return x,y

def pred_load_data(filename):
    df = pd.read_excel(filename)
    data = df.to_numpy()
    return data


def generate_data(x,y):

    np.random.shuffle(x)
    print(x.shape)
    n = x.shape[0]
    idx = int(0.9*n)
    x_train = x[:idx,:]
    y_train = y[:idx]
    x_test = x[idx:,:]
    y_test = y[idx:]

    return x_train,y_train,x_test,y_test

def mean_squared_error(Y, y_pred):
    return tf.reduce_mean(tf.square(y_pred-Y))
def mean_squared_error_deriv(Y,y_pred):
    return tf.reshape(tf.reduce_mean(2*(y_pred-Y)),[1,1])
def h(X,weights,bias):
    return tf.tensordot(X,weights,axes = 1)+bias


x,y = load_data()
x_train,x_test,y_train, y_test = train_test_split(x,y,test_size = 0.2)

trainX = tf.constant(x_train,dtype=tf.float32)
trainY = tf.constant(y_train,dtype = tf.float32)

testX = tf.constant(x_test,dtype = tf.float32)
testY = tf.constant(y_test,dtype = tf.float32)

num_epochs = 30
num_samples = trainX.shape[0]
batch_size = 10
learning_rate = 0.000005

dataset = tf.data.Dataset.from_tensor_slices((trainX,trainY))
dataset = dataset.shuffle(500).repeat(num_epochs).batch(batch_size)

iterator = dataset.__iter__()
num_features = trainX.shape[1]
weights = tf.random.normal((num_features,1))
bias = np.random.randn()

epochs_plot = list()
loss_plot = list()

for i in range(num_epochs):
    epoch_loss = list()
    for b in range(int(num_samples/batch_size)):
        x_batch,y_batch = iterator.get_next()
        output = h(x_batch,weights,bias)
        a = mean_squared_error(y_batch,output).numpy()
        loss = epoch_loss.append(mean_squared_error(y_batch,output).numpy())

        dJ_dH = mean_squared_error_deriv(y_batch,output)
        dH_dW = x_batch
        dJ_dW = tf.reduce_mean(dJ_dH*dH_dW)
        dJ_dB = tf.reduce_mean(dJ_dH)
        weights -= (learning_rate*dJ_dW)
        bias -= (learning_rate*dJ_dB)

    loss = np.array(epoch_loss).mean()
    epochs_plot.append(i+1)
    loss_plot.append(loss)

    print('Epoch {} : loss is {}'.format(i,loss))


output = h( testX , weights , bias )
labels = testY

accuracy_op = tf.metrics.MeanAbsoluteError()
accuracy_op.update_state( labels , output )
print( 'Mean Absolute Error = {}'.format( accuracy_op.result().numpy() ) )


predX = pred_load_data('KFC-dpred2.xlsx')
w = weights.numpy()
predY = np.dot(predX,w)+bias
predY = predY.numpy()
df = pd.DataFrame(predY.copy())
filepath = 'KFC-pred2.xlsx'
df.to_excel(filepath,index = False)
