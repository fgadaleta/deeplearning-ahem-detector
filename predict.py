from __future__ import print_function
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.style as ms
ms.use('seaborn-muted')

from keras.models import model_from_json

import skimage.io as io

from os import listdir
from os.path import isfile, join, isdir
import utils as ut

import librosa
import librosa.display
import argparse

parser = argparse.ArgumentParser(description="Train ahem detector")

parser.add_argument("new_data", action="store")

parser.add_argument("--model", action="store", help="file path for keras json",
					dest="model", default="./ahem_architecture.json")

parser.add_argument("--weights", action="store", help="file path for h5 weights",
					dest="weights", default="./ahem_weights.h5")

config = parser.parse_args()

if not isdir(config.new_data):
	raise Exception("Pass valid a folder path")

# Load the model from disk
with open(config.model, "r") as j:
	model = model_from_json(j.read())

model.load_weights(config.weights)
print("loaded from disk")

model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
newsample_files = [join(config.new_data, f)
				for f in listdir(config.new_data)
				if isfile(join(config.new_data, f))]


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

print("Beginning predictions")
predictions = model.predict_classes(X_test)
print("Finished predictions")

# collect all indices of noisy samples (class 1)
# start position is encoded in filename (a trick to run this in parallel with no sequential order)
noisy_frames = np.where(predictions == 1)[0]
noisy_files = [newsample_files[n] for n in noisy_frames]


# Load a sound with a lot of "ahem" in it
# TODO: Fix file paths - 01/25/17 10:59:36 sidneywijngaarde
sound_file_paths = [join(config.new_data, "..", "provocation_dirty.wav")]
sound_names = ["dirty"]
raw_sounds = ut.load_sound_files(sound_file_paths)
windowsize = 6000
# create positive samples
audiosamples = raw_sounds[0]
numsamples = audiosamples.shape[0]

clean_audio = audiosamples

noisy_start = []
for fn in noisy_files:
	noisy_start.append(int(fn.split('_')[-1].split('.')[0]))

noisy_start.sort(reverse=True)
# collect all indices of noisy samples (class 1)
# start position is encoded in filename (a trick to run this in parallel with no sequential order)
noisy_frames = np.where(predictions == 1)[0]
noisy_files = [newsample_files[n] for n in noisy_frames]

# clean_audio = audiosamples
prev_idx = 0
for start in range(1, len(noisy_start)):
	prev_pos = noisy_start[prev_idx]
	current_pos = noisy_start[start]
	diff = prev_pos - current_pos
	prev_idx += 1

	# set volume to zero for 'ahem' samples
	clean_audio[current_pos:current_pos + windowsize] = 0

librosa.output.write_wav(join(config.new_data, "..", "cleaned.wav"), clean_audio, sr=44100)
