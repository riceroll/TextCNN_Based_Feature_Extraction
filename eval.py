#! /usr/bin/env python
import os
import time
import datetime

import csv
import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn

import data_processor
from textCNN import TextCNN

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("data_dir", "./data/deep_learning_test", "Data source directory.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_string("checkpoint_file", "", "Checkpoint file name")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    x_raw, y_test = data_processor.load_data_labels(FLAGS.data_dir)
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

# Map data into vocabulary
vocabulary_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocabulary")
vocabulary_processor = learn.preprocessing.VocabularyProcessor.restore(vocabulary_path)
x_test = np.array(list(vocabulary_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
if FLAGS.checkpoint_file=="":
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
else:
    checkpoint_file=FLAGS.checkpoint_file
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,  #avoid not finding the chosen device
      log_device_placement=FLAGS.log_device_placement)  #show the device used to run ops & tensors
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        scores = graph.get_operation_by_name("output/scores").outputs[0]
        features = graph.get_operation_by_name("reshape").outputs[0]

        feature_length=sess.run(graph.get_operation_by_name("feature_length").outputs[0])

        # Generate batches for one epoch
        batches = data_processor.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the features
        all_features = np.zeros(feature_length)
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])
            batch_features = sess.run(features, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_features = np.row_stack((all_features,batch_features))
        all_features = all_features[1:]

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# # Save the evaluation to a csv
features_readable = np.column_stack((np.array(x_raw), all_features))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "features.csv")
print("Saving features to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(features_readable)