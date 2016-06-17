# word_segment
Chinese Word Segment demo based on many-many recurrent neural network framework.

Usage:

1. Set your data and training dir in `src/word_seg.py `:
```
# there should be two files in your data dir, train and dev.
# both files should be one sentence per line and tokens of the
# sentence should be seperate by single space.
flags.DEFINE_string("data_dir", 'YOUR_DATA_DIR', "data_dir")

# train dir is used to place the model dumps
flags.DEFINE_string("train_dir", 'YOU_TRAINING_DIR', "train_dir")
```
2. run ```python word_seg.py ```

Notice:

  1. Most of the codes are token from the [Tensorflow RNN example](https://www.tensorflow.org/versions/r0.9/tutorials/recurrent/index.html#recurrent-neural-networks);
  2. There are some sample data in `data/`, use of them should keep to [their liscense](http://sighan.cs.uchicago.edu/bakeoff2005/).
