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

