#! /usr/bin/env python

import os
import time
import datetime

import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn

import data_processor
from textCNN import TextCNN

# Parameters============================================================
# Data
tf.flags.DEFINE_string("data_dir", "./data/deep_learning", "Data source directory.")
tf.flags.DEFINE_float("validation_percentage", 0.1, "Percentage of source data used for validation")
tf.flags.DEFINE_integer("embedding_dim", 128, "Character embedding")
# Hyper parameters
tf.flags.DEFINE_string("filter_sizes", "3,4,5,8,10,20", "Filter sizes")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters of every size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda")
# Training
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epoch (default: 200)")
tf.flags.DEFINE_integer("period_evaluation", 100, "Evaluate model validation set every (100) steps")
tf.flags.DEFINE_integer("period_checkpoint", 100, "Save model every (100) steps")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
for param, value in sorted(FLAGS.__flags.items()):
    print("{}: {}".format(param, value))

# Data============================================================
# Load
x_text, y = data_processor.load_data_labels(FLAGS.data_dir)
# Vocabulary
max_length = max([len(x.split(" ")) for x in x_text])
vocabulary_processor = learn.preprocessing.VocabularyProcessor(max_length)
x = np.array(list(vocabulary_processor.fit_transform(x_text)))
# Randomly shuffle data
np.random.seed(10)
shuffled_sort = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffled_sort]
y_shuffled = y[shuffled_sort]
# Validation set & Training Set
validation_index = -1 * int(FLAGS.validation_percentage * float(len(y)))
x_train, x_validation = x_shuffled[:validation_index], x_shuffled[validation_index:]
y_train, y_validation = y_shuffled[:validation_index], y_shuffled[validation_index:]
print("Vocabulary: {:d}".format(len(vocabulary_processor.vocabulary_)))
print("Train validation ratio: {:d}/{:d}".format(len(y_train), len(y_validation)))

# Training============================================================
with tf.Graph().as_default():
    configuration = tf.ConfigProto(
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=configuration)
    with sess.as_default():
        text_CNN = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocabulary_size=len(vocabulary_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(text_CNN.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Model directory
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
        model_dir = os.path.abspath(os.path.join(os.path.curdir, "models", timestamp))
        print("The model is saved in "+model_dir)

        # Summary directory
        train_summary_dir = os.path.join(model_dir, "summaries", "train")
        validation_summary_dir = os.path.join(model_dir, "summaries", "validation")
                
        # Checkpoint directory
        checkpoint_dir = os.path.abspath(os.path.join(model_dir, "checkpoints"))
        checkpoint_filename = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write summaries
        loss_summary = tf.summary.scalar("loss", text_CNN.loss)
        acc_summary = tf.summary.scalar("accuracy", text_CNN.accuracy)
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        validation_summary_op = tf.summary.merge([loss_summary, acc_summary])
        validation_summary_writer = tf.summary.FileWriter(validation_summary_dir, sess.graph)

        # Write vocabulary
        vocabulary_processor.save(os.path.join(model_dir, "vocabulary"))

        # Initialize the session
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            feed = {
                text_CNN.input_x: x_batch,
                text_CNN.input_y: y_batch,
                text_CNN.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run([train_op, global_step, train_summary_op, text_CNN.loss, text_CNN.accuracy], feed)
            time_str = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
            print("{}: step {}, loss {:g}, accuracy {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def validate_step(x_batch, y_batch, writer=None):
            feed = {
                text_CNN.input_x: x_batch,
                text_CNN.input_y: y_batch,
                text_CNN.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run([global_step, validation_summary_op, text_CNN.loss, text_CNN.accuracy], feed)
            time_str = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
            print("{}: step {}, loss {:g}, accuracy {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_processor.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.period_evaluation == 0:
                print("\nEvaluation:")
                validate_step(x_validation, y_validation, writer=validation_summary_writer)
                print("")
            if current_step % FLAGS.period_checkpoint == 0:
                path = saver.save(sess, checkpoint_filename, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
