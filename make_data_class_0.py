import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import os, sys
import utils as ut

# Load sound files
path = '/archive1/ahem_data/'
sound_file_paths = [os.path.join(path, "ahem_sounds.wav"),
                    os.path.join(path, "podcast_17_sample.wav"),
                   ]

sound_names = ["ahem_sounds", "podcast_17_sample"]
raw_sounds = ut.load_sound_files(sound_file_paths)

windowsize = 6000  # size of sliding window (22050 samples == 0.5 sec)  
step       = 3000
maxfiles   = 10000
numfiles   = 0

dimx = 6
dimy = 5

# create negative samples
audiosamples = raw_sounds[1]
numsamples = audiosamples.shape[0]

for x in xrange(0, numsamples-windowsize, step):
    numfiles += 1 
    b = x               # begin 
    e = b+windowsize    # end 
    ut.printStuff('Creating spectrum image class_0 samples [%d-%d] of %d file %d',(b,e, numsamples, numfiles))
    filename = os.path.join(path, 'class_0/partial_spectrum_%d.png'%x)
    ut.specgram_frombuffer(audiosamples[b:e], dimx, dimy, fname=filename, dpi=180)
    #if x == maxfiles:
    #    break
        
print('\nbye!\n')        