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

import os, sys
import utils as ut

# Load sound files
path = '/data'
sound_file_paths = [os.path.join(path, "provocation_dirty.wav"),
                    os.path.join(path, "provocation_clean.wav"),
                   ]

sound_names = ["dirty", "clean"]
raw_sounds = ut.load_sound_files(sound_file_paths)

windowsize = 6000  # size of sliding window (22050 samples == 0.5 sec)  
step       = 3000
maxfiles   = 10000

dimx = 6
dimy = 5

# create positive samples
audiosamples = raw_sounds[0]
numsamples = audiosamples.shape[0]
numfiles = 0

for x in xrange(0, numsamples-windowsize, step):
    numfiles += 1 
    b = x               # begin 
    e = b+windowsize    # end 
    ut.printStuff('Creating spectrum image new samples [%d-%d] of %d file %d',(b,e, numsamples, numfiles))
    filename = os.path.join(path, '/archive/ahem_data/new_sample/partial_spectrum_%d.png'%x)
    ut.specgram_frombuffer(audiosamples[b:e], dimx, dimy, fname=filename, dpi=180)
    
        
print('\nbye!\n')        