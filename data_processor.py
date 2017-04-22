import os

import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    # Tokenization 
    
    string = re.sub(re.compile("<math>.*?</math>"), " ", string)
    string = re.sub(re.compile("<url>.*?</url>"), " ", string)
    string = re.sub(re.compile("<.*?>"), " ", string)
    string = re.sub(re.compile("&.*?;"), " ", string)
    string = re.sub(re.compile("/.*?>"), " ", string)
    string = re.sub(re.compile("i>"), " ", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)

    return string.strip().lower()

def load_data_labels(data_dir):
    # Loads texts of different labels form files
    # Returns split sentences and labels.

    x = [open(os.path.join(os.path.abspath(data_dir),i),"rb").readlines() for i in os.listdir(os.path.abspath(data_dir)) if i[0]!='.']
    # print(open(os.path.join(os.path.abspath(data_dir),"test.neg"),"rb").readlines())
    x = [[str(k,encoding="utf-8").strip() for k in s] for s in x]
    x = [[clean_str(k) for k in s] for s in x]

    x_text = []
    tmp = []
    y = []
    for i in x:
        for j in i:
            x_text.append(j)
    for i in range(len(x)):
        for j in x[i]:
            tmp.append(i)
    y=[[0 for j in x] for i in x_text]
    for i in range(len(tmp)):
        y[i][tmp[i]]=1
    y=np.array(y)

    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    # Generates a batch iterator for a dataset.

    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
