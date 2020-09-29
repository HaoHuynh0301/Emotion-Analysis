import pandas as pd
import numpy as np
import matplotlib as mpl
import tensorflow as tf
from keras.utils import np_utils
from keras.regularizers import l2
import sys
import os
import cv2
import keras

data=pd.read_csv('fer2013.csv')
# print(data["Usage"].value_counts())
# print(data)

x_train, y_train, x_test, y_test=[],[],[],[]

for index, row in data.iterrows():
    val=row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
            x_train.append(np.array(val, 'float32'))
            y_train.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
            x_test.append(np.array(val, 'float32'))
            y_test.append(row['emotion'])

    except:
        print("Error!!!")

num_features=64
num_label=7
batch_size=64
epochs=30
width, height=48, 48

x_train=np.array(x_train, 'float32')
y_train=np.array(y_train, 'float32')
x_test=np.array(x_test, 'float32')
y_test=np.array(y_test, 'float32')

y_train=np_utils.to_categorical(y_train, num_classes = num_label)
y_test=np_utils.to_categorical(y_test, num_classes = num_label)


x_train-=np.mean(x_train, axis = 0)
x_train/=np.std(x_train, axis = 0)

x_test-=np.mean(x_test, axis = 0)
x_test/=np.std(x_test, axis = 0)

x_train=x_train.reshape(x_train.shape[0], width, height, 1)
x_test=x_test.reshape(x_test.shape[0], width, height, 1)

model=keras.Sequential()

model.add(keras.layers.Conv2D(num_features, kernel_size = (3,3), activation = 'relu', input_shape = (x_train.shape[1:])))
model.add(keras.layers.Conv2D(num_features, kernel_size = (3,3), activation = 'relu'))
model.add(keras.layers.MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(keras.layers.Dropout(0.5))


model.add(keras.layers.Conv2D(num_features, kernel_size = (3,3),activation = 'relu'))
model.add(keras.layers.Conv2D(num_features, kernel_size = (3,3), activation = 'relu'))
model.add(keras.layers.MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(keras.layers.Dropout(0.5))


model.add(keras.layers.Conv2D(2*num_features, kernel_size = (3,3),activation = 'relu'))
model.add(keras.layers.Conv2D(2*num_features, kernel_size = (3,3), activation = 'relu'))
model.add(keras.layers.MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(2*2*2*2*num_features, activation = 'relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(2*2*2*2*num_features, activation = 'relu'))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(num_label, activation = 'softmax'))

model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adam(), metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs = epochs, verbose = 1, validation_data = (x_test, y_test))
model.save('model.h5')