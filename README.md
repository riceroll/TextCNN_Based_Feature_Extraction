# CNN_Person_Reidentification
Re-identify pedestrians with CNN.

## Task
Extract the feature of a text.

## Requirements

- Python 3
- Tensorflow
- Numpy

## Training

Training parameters:

```bash
./train.py --help
``````

Train:

```bash
./train.py
``````

## Evaluating

```bash
./eval.py --eval_train --checkpoint_dir="./runs/1459637919/checkpoints/"
``````

Replace the checkpoint dir with the output from the training. To use your own data, change the `eval.py` script to load your data.


## References

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf)
- [Tensorflow Module: tf.contrib.learn](https://www.tensorflow.org/api_docs/python/tf/contrib/learn)
