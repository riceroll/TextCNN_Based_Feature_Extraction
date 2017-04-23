# CNN_Person_Reidentification
Re-identify pedestrians with CNN.

## Introduction
### Dataset
#### RIDLT 
A dataset created for person re-identification task, including 274 identities, 38934 bounding boxes, created by the lab of Professor [Hua Yang](http://icne.sjtu.edu.cn/info/1059/1073.htm) of Shanghai Jiao Tong University.
#### Image_Clipper 
A tool helps to clip bounding boxes from original images from video cameras. The code of the tool is [here](https://github.com/riceroll/image_clipper).
### Training
Train the model with a CNN modified from Caffenet, whose task is classification. The sizes of fc6 and fc7 are modified into 1024 in order to avoid overfitting. The size of fc8 is modified into 151, which is the number of identities in the training set.
### Feature Extraction
Abandong fc8 and extract the vector of fc7 as the feature of an image. Save the features of query sequence and library sequence into a .mat file.
### Evaluation
Adopt XQDA as metric learning method. Use CMC curve to show the performance. The code of evaluation is [here](https://github.com/riceroll/Evaluation_Person_Reidentification).

## Requirements

- Python 3
- Tensorflow
- Numpy

## Quick Start
### Download Data
Since the test set and training set are uploaded to MEGA.
Get the deploy file, models and test set by typing this command in the root directory:
```bash
sh 
```

### Training
Get the training options by typing this command:
```bash
./train.py --help
```

Train:
```bash
./train.py
```

### Evaluating
```bash
run eval.py --eval_train --checkpoint_dir="./models/2017-04-22_01-46-18/checkpoints/" --checkpoint_file="./models/2017-04-22_01-46-18/checkpoints/model-900"
```

Replace the checkpoint dir with the output from the training. To use your own data, change the `eval.py` script to load your data.



## References

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf)
- [Tensorflow Module: tf.contrib.learn](https://www.tensorflow.org/api_docs/python/tf/contrib/learn)
