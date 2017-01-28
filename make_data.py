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
"""

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

from os import path, listdir, mkdir
import utils as ut
import argparse

parser = argparse.ArgumentParser(description="Create data for ahem detector")

parser.add_argument("data_dir", action="store")

config = parser.parse_args()
print("Creating samples from: {}".format(config.data_dir))

if not path.isdir(config.data_dir):
	raise Exception("First Argument is not a directory")

# Load sound files
sound_files = [path.join(config.data_dir, f)
				for f in listdir(config.data_dir)
				if f.endswith("wav")]

if len(sound_files) == 0:
	raise Exception("There are no wav files in path: {}".format(config.data_dir))

raw_sounds = ut.load_sound_files(sound_files)

image_path = path.join(config.data_dir, "images")
if not path.isdir(image_path):
	mkdir(image_path)

windowsize = 6000  # size of sliding window (22050 samples == 0.5 sec)
step = 3000
numfiles = 0

dimx = 6
dimy = 5

for i in range(len(raw_sounds)):
	# create samples
	numsamples = raw_sounds[i].shape[0]
	file_path = path.basename(sound_files[i])
	file_path = path.splitext(file_path)[0]
	for x in range(0, numsamples - windowsize, step):
		b = x               # begin
		e = x + windowsize  # end

		fmt_string = "(%d/%d) %s [%d-%d] of %d file %d"
		ut.printStuff(fmt_string, (i, len(raw_sounds) - 1, file_path, x, e, numsamples, numfiles))

		filename = path.join(image_path, "{}_{}.png".format(file_path, x))
		ut.specgram_frombuffer(raw_sounds[i][x:e], dimx, dimy, fname=filename, dpi=180)

		numfiles += 1

print('\nbye!\n')
