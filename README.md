# Deep Learning 'ahem' detector #

![alt text](https://github.com/worldofpiggy/deeplearning-ahem-detector/raw/master/ahem_explained.PNG "Ahem neural detector explained")

The ahem detector is a deep convolutional neural network that is trained on transformed audio signals to recognize "ahem" sounds.
The network has been trained to detect such signals on the episodes of Data Science at Home, the podcast about data science at 
[worldofpiggy.com/podcast](http://worldofpiggy.com/podcast) 

Slides and some technical details provided [here](https://docs.google.com/presentation/d/1QXQEOiAMj0uF2_Gafr2bn-kMniUJAIM1PLTFm1mUops/edit?usp=sharing).

Two sets of audio files are required, very similarly to a cohort study:

- a negative sample with clean voice/sound and 

- a positive one with "ahem" sounds concatenated

While the detector works for the aforementioned audio files, it can be generalized to any other audio input, provided enough data 
are available. The minimum required is ~10 seconds for the positive samples and ~3 minutes for the negative cohort. 
The network will adapt to the training data and can perform detection on different spoken voice.


### How do I get set up? ###
Once the training audio files are provided, just load the training set and train the network with the code in the ipython notebook.
Make sure to create the local folder that has been hardcoded in the script files below.
Build training/testing set before running the script. 
Execute first 

    % python make_data_class_0.py
    % python make_data_class_1.py

A GPU is recommended as, under the conditions specific to this example at least 5 epochs are required to obtain ~81% accuracy.


### How do I clean a new dirty audio file?
A new audio file must be trasformed in the same way of training files.
This can be done with

    % python make_data_newsample.py
    
Then follow the script in the ipython notebook that is commented enough to proceed without particular issues.







## License and Copyright Notice

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
