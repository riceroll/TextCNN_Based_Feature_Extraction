# TextCNN_Based_Feature_Extraction
Extract the features of the abstracts of academic papers

## Introduction
Use CNN to extract the features of academic texts. The feafures can be used to calculate the similarity between papers, which can help fulfill the rating matrix generated by reference relationship. Then, collaborative filtering recommender can use the fulfilled rating matrix to do the recommendation. The code of collaborative filtering is [here](https://github.com/riceroll/Collaborative_Filtering_Recommender).

### Dataset
Academeic articles from fields of computer vision, natural language processing and speech recognition, which is crawled from IEEE and ACM.

## Requirements

- Python>3.5
- Tensorflow
- Numpy

## Quick Start
### Download Data
Since the test set and training set are uploaded to MEGA.
Get the models and dataset by typing this command in the root directory:
```bash
sh download_data.sh
```

### Training
Get the training options by typing this command:
```bash
python train.py --help
```

Train:
```bash
python train.py
```

### Evaluating
```bash
python eval.py --eval_train --checkpoint_dir="./models/2017-04-22_01-46-18/checkpoints/" --checkpoint_file="./models/2017-04-22_01-46-18/checkpoints/model-900"
```

Replace the checkpoint dir with the output from the training. To use your own data, change the `eval.py` script to load your data.



## References

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf)
- [Tensorflow Module: tf.contrib.learn](https://www.tensorflow.org/api_docs/python/tf/contrib/learn)
