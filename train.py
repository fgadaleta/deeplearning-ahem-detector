# coding: utf-8
"""
MIT License
Copyright (c) 2016 Francesco Gadaleta

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

--------------------------------------------------------------------------------------
Note:
Please build training/testing set before running this script.
Make sure to create the local path that have been hardcoded in the following scripts,
then execute

	% python make_data_class_0.py
	% python make_data_class_1.py
---------------------------------------------------------------------------------------

"""

from __future__ import print_function
import numpy as np

import matplotlib
matplotlib.use('Agg')
# Force matplotlib to not use any Xwindows backend.
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import matplotlib.style as ms
ms.use('seaborn-muted')
# get_ipython().magic('matplotlib inline')

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json
# from keras.utils.visualize_util import plot

import skimage.io as io
# from skimage.measure import block_reduce

from os import listdir
from os.path import isfile, join, isdir

import argparse

def make_model():
	model = Sequential()
	print(len(kernel_size))
	model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
							border_mode='valid',
							input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], dim_ordering="th"))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size))
	model.add(Dropout(0.25))

	model.add(Convolution2D(nb_filters, 3, 3, border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filters, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	model.compile(loss='binary_crossentropy',
					optimizer='adadelta',
					metrics=['accuracy'])
	return model

def load_image(filename):
	img = io.imread(filename)
	img = img.transpose((2, 0, 1))
	img = img[:3, :, :]
	return img

if __name__ == "__main__":
	# np.random.seed(1337)  # for reproducibility

	# network configuration
	batch_size = 32
	# number of epochs
	nb_epoch = 5
	# number of convolutional filters to use
	nb_filters = 32
	# number of classes
	nb_classes = 2
	# size of pooling area for max pooling
	pool_size = (2, 2)
	# convolution kernel size
	kernel_size = (3, 3)

	parser = argparse.ArgumentParser(description="Train ahem detector")
	parser.add_argument("class_0", action="store")
	parser.add_argument("class_1", action="store")

	config = parser.parse_args()

	if not isdir(config.class_0) or not isdir(config.class_1):
		raise Exception("Pass valid folder paths")

	class_0_img_dir = join(config.class_0, "images")
	class_1_img_dir = join(config.class_1, "images")

	if not isdir(class_0_img_dir) or not isdir(class_1_img_dir):
		raise Exception("Paths do not contain images subdirectory")

	# Load sound files
	class_0_files = [join(class_0_img_dir, f)
					for f in listdir(class_0_img_dir)
					if isfile(join(class_0_img_dir, f))]

	class_1_files = [join(class_1_img_dir, f)
					for f in listdir(class_1_img_dir)
					if isfile(join(class_1_img_dir, f))]

	print("Class_0 files:", len(class_0_files))
	print("Class_1 files:", len(class_1_files))

	# prepare training set
	X_t = []
	Y_t = []

	for fn in class_0_files[:400]:
		img = io.imread(fn)
		img = img.transpose((2, 0, 1))
		img = img[:3, :, :]
		X_t.append(img)
		Y_t.append(0)

	for fn in class_1_files[:40]:
		img = io.imread(fn)
		img = img.transpose((2, 0, 1))
		img = img[:3, :, :]
		X_t.append(img)
		Y_t.append(1)

	X_t = np.asarray(X_t)
	X_t = X_t.astype('float32')
	X_t /= 255

	Y_t = np.asarray(Y_t)
	Y_t = np_utils.to_categorical(Y_t, nb_classes)

	img_rows, img_cols = X_t.shape[2], X_t.shape[3]
	# input image dimensions
	img_channels = 3			   # RGB
	input_shape = (3, img_rows, img_cols)

	# test set
	X_test = []
	Y_test = []

	for fn in class_0_files[400:len(class_0_files)]:
		img = io.imread(fn)
		img = img.transpose((2, 0, 1))
		img = img[:3, :, :]
		X_test.append(img)
		Y_test.append(0)

	for fn in class_1_files[40:len(class_1_files)]:
		img = io.imread(fn)
		img = img.transpose((2, 0, 1))
		img = img[:3, :, :]
		X_test.append(img)
		Y_test.append(1)

	X_test = np.asarray(X_test)
	Y_test = np.asarray(Y_test)
	X_test = X_test.astype('float32')
	X_test /= 255

	Y_test = np_utils.to_categorical(Y_test, nb_classes)

	# model = make_model()
	with open("./models/ahem_architecture.json", "r") as j:
		model = model_from_json(j.read())

	model.load_weights("./models/ahem_weights.h5")
	print("loaded from disk")

	model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

	for _ in range(3):
		model.fit(X_t, Y_t,
				# validation_data=(X_test, Y_test),
				batch_size=batch_size,
				nb_epoch=1, verbose=1)

	predictions = model.predict_classes(X_test)

	y = []
	for e in Y_test:
		if e[0] > e[1]:
			y.append(0)
		else:
			y.append(1)

	correct = np.sum(y == predictions)
	percent = (correct / Y_test.shape[0]) * 100
	print('{}/{}'.format(correct, Y_test.shape[0]), "{}%".format(percent))

	model_json = model.to_json()
	with open("./my_model.json", "w") as json_file:
		json_file.write(model_json)
	model.save_weights("./my_model_weights.h5")

	print("Model written to:", "my_model.json", "my_model_weights.h5")
