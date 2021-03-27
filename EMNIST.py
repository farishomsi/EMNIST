
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# we will choose the digits dataset since we want a classification of digits


train = pd.read_csv('C:Users/ACER/Desktop/files-python/emnist-digits-train.csv', delimiter=',')
test = pd.read_csv('C:Users/ACER/Desktop/files-python/emnist-digits-test.csv', delimiter=',')
mapp = pd.read_csv('C:Users/ACER/Desktop/files-python/emnist-digits-mapping.txt', delimiter=',')
# print(train.head())
train.info()
# since we dont have much of a featurs here we should try at least to come up with the images that we are classifying
# but first let us define our vaiables and input / split the data
trainData = train.values
testData = test.values
# . values gives back the numpy array so we dont mix it with the panda series
x_train = trainData[:, 1:].astype('float32')
y_train = trainData[:, 0:1]

x_test = testData[:, 1:].astype('float32')
y_test = testData[:, 0:1]
# normalizing the data
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# lets try to plot the image
j = 10
m = x_train.shape[0]
X = x_train.reshape((m, 28, 28, 1))
# plt.figure(1)
# plt.gray()
# plt.imshow(X[j].reshape((28,28)))
# plt.show()
# preprocessing
# first we have no null values
# we need to normalize the data using one hot encoding or scale the data
# scalling data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# now we create a model and train it
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout

model = Sequential()
model.add(Dense(units=784, activation='relu'))
model.add(Dense(units=784, activation='relu'))
model.add(Dense(units=500, activation='relu'))
model.add(Dense(units=250, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# we will use a dense of 784


print(x_train.shape)
# to avoid overfitting we will use two methods to avoid that
# 1 early stopping
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
model.fit(x=x_train,
          y=y_train,
          epochs=600,
          validation_data=(x_test, y_test), verbose=1,
          callbacks=[early_stop]
          )

model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
plt.show()






"""" 

// // // // // // / the
updated
version
without
early
stop and picture """

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

import os

batch_size = 32
# we will try batch size = 64 >>
# epoch = 25 >> acc = 0.943 , loss = 0.189
# epochs = 50 >> acc = 0.957 , loss = 0.1385
# epochs = 75  >> acc = 0.9668 , loss = 0.110
# batch size = 32
# epochs 25 >>> acc = 0.950 , loss = 0.164
# epochs = 50 >> acc = 0.964 , loss = 0.119
# epochs = 75  >> acc =0.970 , ; loss = 0.0982
num_classes = 10
# 0-9
epochs = 25
# with 10 >>> acc = 0.903 , loss = 0.340
# with 15 >>> acc =0.919 , loss = 0.275
# with 18 >> acc = 0.93 , loss = 0.252
# with 22 >> acc = 0.9335 , loss = 0.233
# with 27 >> acc = 0.939 , loss = 0.21
# with 35 >> acc = 0.94 , loss = 0.188
# with 45 >> acc = 0.95 , loss = 0.167
# with 70 >>> acc = 0.959 , loss = 0.136


digits_train_data = pd.read_csv('C:Users/ACER/Desktop/files-python/emnist-digits-train.csv', delimiter=',')
digits_test_data = pd.read_csv('C:Users/ACER/Desktop/files-python/emnist-digits-test.csv', delimiter=',')

print('data loaded')

datas_train = digits_train_data.values
datas_test = digits_test_data.values
# split the data
x_train = datas_train[:, 1:].astype('float32')
y_train = datas_train[:, 0:1]

x_test = datas_test[:, 1:].astype('float32')
y_test = datas_test[:, 0:1]
# normalize the data
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# transform to a binary class matrix so we can be able to use catogorical cross entropy
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
# we choose 4 denses
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'), )
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'), )
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

# compiling the model
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# training the model

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_split=0.1)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


