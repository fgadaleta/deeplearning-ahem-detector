## Pitchback Speech Recognition
forked from
[deeplearning-ahem-detector](reverse=Tru://github.com/worldofpiggy/deeplearning-ahem-detector)

[Original Documentation](./ahem.md)

## Getting Started
It is recommended to run this in a virtual environment

```bash
$ virtualenv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

1. Generate training data
To generate training data use [make_data.py](./make_data.py). Ahem detector is
a binary classifier and therefore expects two classes for training.

The directory should contain wav files of the same class

```bash
$ python make_data.py [directory]
```

Perform the above on __data/ahem_data__ and __data/clean_data__ to start off.
eg.
```bash
$ python make_data.py data/ahem_data/
$ python make_data.py data/clean_data/
```

2. Train the model
Once the images have been created we can train the model. Pass the same file
path as used in [make_data.py](./make_data.py) above. Each should now contain a
nested image directory.

```bash
$ python train [class_0 dir] [class_1 dir]
```

3. Predict on new samples
Use the model weights from the previous step to predict the timestamps of "uhh"
in a new file.

```bash
$ python make_data.py [directory]
$ python predict.py [directory] [--model MODEL] [--weights WEIGHTS] 
```

Use the --model and --weights flags to load in the model outputted in step 2.
Predict will use [models/model.json](models/model.json) and
[models/weights.h5](models/weights.h5) by default.

The output will be written to a json file in the directory passed
