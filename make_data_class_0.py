import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import os, sys
import utils as ut

# Load sound files
path = './data'
sound_file_paths = [os.path.join(path, "ahem_sounds.wav"),
                    os.path.join(path, "podcast_17_sample.wav"),
                   ]

sound_names = ["ahem_sounds", "podcast_17_sample"]
raw_sounds = ut.load_sound_files(sound_file_paths)

windowsize = 44100  # size of sliding window (22050 samples == 0.5 sec)  
step       = 22050
maxfiles   = 50000

# create negative samples
audiosamples = raw_sounds[1]
numsamples = audiosamples.shape[0]
for x in xrange(numsamples - windowsize-step):
    ut.printStuff('Creating spectrum image class_0 %d of %d', (x, maxfiles))
    b = x+step            # begin 
    e = x+windowsize      # end 
    filename = os.path.join(path, 'class_0/partial_spectrum_%d.png'%x)
    ut.specgram_frombuffer(audiosamples[b:e], 5, 2, fname=filename, dpi=96)
    if x == maxfiles:
        break
        
print('bye!\n')        