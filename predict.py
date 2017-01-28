from __future__ import print_function
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.style as ms
ms.use('seaborn-muted')

from keras.models import model_from_json

import skimage.io as io

from os import listdir
from os.path import join, isdir
import utils as ut
import json

import librosa
import librosa.display
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train ahem detector")

	parser.add_argument("new_data", action="store")

	parser.add_argument("--model", action="store", help="file path for keras json",
						dest="model", default="./models/model.json")

	parser.add_argument("--weights", action="store", help="file path for h5 weights",
						dest="weights", default="./models/weights.h5")

	config = parser.parse_args()

	if not isdir(config.new_data):
		raise Exception("Pass valid a folder path")

	imgDir = join(config.new_data, "images")
	if not isdir(imgDir):
		raise Exception("{} does not contain images subdirectory" % config.new_data)

	# Load the model from disk
	with open(config.model, "r") as j:
		model = model_from_json(j.read())

	model.load_weights(config.weights)
	print("Loaded:", config.model, config.weights)

	model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
	newsample_files = [join(imgDir, f) for f in listdir(imgDir)]

	# prepare test set as we did for training set
	X_test = []

	for fn in newsample_files:
		img = io.imread(fn)
		img = img.transpose((2, 0, 1))
		img = img[:3, :, :]
		X_test.append(img)

	X_test = np.asarray(X_test)
	X_test = X_test.astype('float32')
	X_test /= 255

	predictions = model.predict_classes(X_test)

	# collect all indices of uhh (class 1)
	noisy_frames = np.where(predictions == 1)[0]
	noisy_files = [newsample_files[n] for n in noisy_frames]

	noisy_start = []
	for fn in noisy_files:
		# Start position is encoded in filename
		# Can use this to run in parallel
		noisy_start.append(int(fn.split('_')[-1].split('.')[0]))

	noisy_start.sort(reverse=True)

	# NOTE: There should only be one wav file here
	sound_file = [join(config.new_data, f)
					for f in listdir(config.new_data)
					if f.endswith("wav")]

	raw_sounds = ut.load_sound_files(sound_file)

	# create positive samples
	audiosamples = raw_sounds[0]
	numsamples = audiosamples.shape[0]

	prev_idx = 0
	windowsize = 6000

	# 22050 samples == 0.5 sec
	def frameToSec(startFrame):
		start = (startFrame * 0.5) / 22050
		return [start, start + 0.068]

	# Convert to seconds
	seconds = [frameToSec(i) for i in noisy_start]

	outPath = join(config.new_data, "results.json")
	with open(outPath, "w") as outFile:
		json.dump({"timestamps": seconds[::-1]}, outFile, indent=2)
		print("Results written to:", outPath)

	# TODO: Remove from here down once we have a certain degree of confidence.
	# We only care about the timestamps - 01/28/17 15:14:36 sidneywijngaarde

	# Silence the section of the audio file
	for start in range(len(noisy_start)):
		current_pos = noisy_start[start]

		# set volume to zero for 'ahem' samples
		audiosamples[current_pos:current_pos + windowsize] = 0

	librosa.output.write_wav(join(config.new_data, "cleaned.wav"), audiosamples, sr=44100)
