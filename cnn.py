'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import skimage.io as io 
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import os 

batch_size = 16
nb_epoch = 12

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
path_class_0 = './data/class_0/'
path_class_1 = './data/class_1/'

# input image dimensions
img_rows, img_cols = 158, 164
nb_classes = 2
input_shape = (1, img_rows, img_cols)

class0_files = [f for f in listdir(path_class_0) if isfile(join(path_class_0, f))]
class1_files = [f for f in listdir(path_class_1) if isfile(join(path_class_1, f))]


X_t = []
Y_t = []

for fn in class0_files[:1000]:
    img = io.imread(os.path.join(path_class_0, fn), as_grey=True)
    X_t.append([img])
    Y_t.append(0)

for fn in class1_files[:1000]:
    img = io.imread(os.path.join(path_class_1, fn), as_grey=True)
    X_t.append([img])
    Y_t.append(1)


X_t = np.asarray(X_t)
Y_t = np.asarray(Y_t)
X_t = X_t.astype('float32')
X_t /= 255



model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('softmax'))


model.compile(loss='binary_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_t, Y_t, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)
