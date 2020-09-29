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

x_train=np.array(x_train, 'float32')
y_train=np.array(y_train, 'float32')
x_test=np.array(x_test, 'float32')
y_test=np.array(y_test, 'float32')

x_train-=np.mean(x_train, axis = 0)
x_train/=np.std(x_train, axis = 0)

x_test-=np.mean(x_test, axis = 0)
x_test/=np.std(x_test, axis = 0)

num_features=64
num_label=7
batch_size=64
epochs=50
width, height=48, 48

x_train=x_train.reshape(x_train.shape[0], width, height, 1)
x_test=x_test.reshape(x_test.shape[0], width, height, 1)
