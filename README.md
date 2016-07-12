# word_segment
Chinese Word Segment demo based on many-many recurrent neural network framework.

Usage
1. Set your data and training dir in `src/word_seg.py `
```
# there should be two files in your data dir, train and dev.
# both files should be one sentence per line and tokens of the
# sentence should be seperated by a single space.
flags.DEFINE_string("data_dir", 'YOUR_DATA_DIR', "data_dir")

# train dir is used to place the model dumps
flags.DEFINE_string("train_dir", 'YOU_TRAINING_DIR', "train_dir")
```
2. run ```python word_seg.py ```
 
Result:

Experiment setup: I  mixed ```WordSeg/data/icwb2-data/training/pku_training.utf8 ``` and ```WordSeg/data/icwb2-data/training/msr_training.utf8 ``` as the training data, mixed their counterparts in ```WordSeg/data/icwb2-data/gold (pku_test_gold.utf8 and msr_test_gold.utf8)``` as test data.

Training: I stopped after 64035 steps

eval:
```
do testing on test data...
Test Perplexity: 1.341
precision:  0.854580714994 , recall:  0.839846347049
```

Notice:

  1. Most of the codes are taken from the [Tensorflow RNN example](https://www.tensorflow.org/versions/r0.9/tutorials/recurrent/index.html#recurrent-neural-networks);
  2. There are some sample data in `data/`, use of them should keep to [their liscense](http://sighan.cs.uchicago.edu/bakeoff2005/).
