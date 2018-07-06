from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
from numpy.random import permutation

from grid_data import GridData

from compute_mpe import CircuitMPE

import tensorflow as tf


FLAGS = None


def weight_variable(shape):
  return tf.Variable(tf.truncated_normal(shape, 0.1))

def bias_variable(shape):
  return tf.Variable(tf.truncated_normal(shape, 0.1))

def main(_):
  # Import data
  # mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  grid_data = GridData(FLAGS.data)

  # Create the model
  # Input(16) - Layer 1(units)
  x = tf.placeholder(tf.float32, [None, 40])
  W = []
  b = []
  ys = []
  W.append(weight_variable([40, FLAGS.units]))
  b.append(bias_variable([FLAGS.units]))
  if FLAGS.relu:
    ys.append(tf.nn.relu(tf.matmul(x, W[0]) + b[0]))
  else:
    ys.append(tf.nn.sigmoid(tf.matmul(x, W[0]) + b[0]))

  for i in range(1, FLAGS.layers):
    # Layer i(units) - Layer i+1(units)
    W.append(weight_variable([FLAGS.units, FLAGS.units]))
    b.append(bias_variable([FLAGS.units]))
    if FLAGS.relu:
      ys.append(tf.nn.relu(tf.matmul(ys[i-1], W[i]) + b[i]))
    else:
      ys.append(tf.nn.sigmoid(tf.matmul(ys[i-1], W[i]) + b[i]))

  # Layer n(units) - Output(24)
  W.append(weight_variable([FLAGS.units, 24]))
  b.append(bias_variable([24]))
  y = tf.matmul(ys[-1], W[-1]) + b[-1] + np.finfo(float).eps * 10


  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 24])
  yu = tf.unstack(tf.nn.sigmoid(y), axis=1)
  xu = tf.unstack(x, axis=1)

  # Create CircuitMPE instance for predictions
  cmpe = CircuitMPE('4-grid-out.vtree.sd', '4-grid-all-pairs-sd.sdd')
  wmc = cmpe.get_tf_ac([[1.0 - ny,ny] for ny in yu + xu[24:]])

  # Get supervised part (rest is unsupervised)
  perm = permutation(grid_data.train_data.shape[0])
  sup_train_inds = perm[:int(grid_data.train_data.shape[0] * FLAGS.give_labels)]
  unsup_train_inds = perm[int(grid_data.train_data.shape[0] * FLAGS.give_labels):]
  ce_weights = np.zeros([grid_data.train_data.shape[0], 1])
  ce_weights[sup_train_inds, :] = 1
  # cross_entropy = tf.reduce_mean(
  #     tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))
  cross_entropy = tf.losses.sigmoid_cross_entropy(y_, y, weights=ce_weights)
  regularizers = sum(tf.nn.l2_loss(weights) for weights in W)
  if FLAGS.use_unlabeled:
    loss = cross_entropy - FLAGS.wmc * tf.log(tf.reduce_mean(wmc)) + FLAGS.l2_decay * regularizers
  else:
    loss = cross_entropy - FLAGS.wmc * tf.log(tf.reduce_mean(wmc * ce_weights)) + FLAGS.l2_decay * regularizers
  # train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  train_step = tf.train.AdamOptimizer().minimize(loss)

  full_loss = tf.losses.sigmoid_cross_entropy(y_, y) - FLAGS.wmc * tf.log(tf.reduce_mean(wmc))

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  # For early stopping
  prev_loss = 1e15
  # Train
  for i in range(FLAGS.iters):
    # batch_xs, batch_ys = grid_data.get_batch(32)
    batch_xs, batch_ys = grid_data.train_data, grid_data.train_labels
    # batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    # Every 1k iterations check accuracy
    if i % 100 == 0:
      print("After %d iterations" % i)
      # Get outputs
      train_out = sess.run(tf.nn.sigmoid(y), feed_dict={x: grid_data.train_data[sup_train_inds, :],
                                      y_: grid_data.train_labels[sup_train_inds, :]})
      valid_out = sess.run(tf.nn.sigmoid(y), feed_dict={x: grid_data.valid_data,
                                      y_: grid_data.valid_labels})
      # Percentage that are exactly right
      print("Percentage of training that are exactly right: %f" % (sum(1 for z in np.sum(np.abs(np.array(train_out + 0.5, int) - grid_data.train_labels[sup_train_inds, :]), axis=1) if z == 0)/float(sup_train_inds.shape[0])))
      print("Percentage of validation that are exactly right: %f" % (sum(1 for z in np.sum(np.abs(np.array(valid_out + 0.5, int) - grid_data.valid_labels), axis=1) if z == 0)/float(grid_data.valid_labels.shape[0])))

      # Percentage of individual labels that are right
      print("Percentage of individual labels in training that are right: %f" % (1. - np.sum(np.abs(np.array(train_out + 0.5, int) - grid_data.train_labels[sup_train_inds, :]))/float(sup_train_inds.shape[0] * grid_data.train_labels.shape[1])))
      print("Percentage of individual labels in validation that are right: %f" % (1. - np.sum(np.abs(np.array(valid_out + 0.5, int) - grid_data.valid_labels))/float(grid_data.valid_labels.shape[0] * grid_data.valid_labels.shape[1])))

      # MPE instatiation accuracy
      mpe_pred = np.array([cmpe.compute_mpe_inst([(1-p,p) for p in np.concatenate((o,inp[24:]))])[:24] for o, inp in zip(train_out, grid_data.train_data[sup_train_inds, :])])
      print("Train MPE accuracy %f" % (float(np.sum(np.equal(np.sum(np.abs(mpe_pred - grid_data.train_labels[sup_train_inds, :]), axis = 1), 0))) / float(sup_train_inds.shape[0])))
      valid_mpe_pred = np.array([cmpe.compute_mpe_inst([(1-p,p) for p in np.concatenate((o, inp[24:]))])[:24] for o, inp in zip(valid_out, grid_data.valid_data)])
      print("Validation MPE accuracy %f" % (float(np.sum(np.equal(np.sum(np.abs(valid_mpe_pred - grid_data.valid_labels), axis = 1), 0))) / float(grid_data.valid_data.shape[0])))

      # Print Losses and WMC
      print("Supervised train loss: %f" % sess.run(full_loss, feed_dict={x: grid_data.train_data[sup_train_inds, :], y_: grid_data.train_labels[sup_train_inds, :]}))
      print("Train loss: %f" % sess.run(full_loss, feed_dict={x: grid_data.train_data, y_: grid_data.train_labels}))
      valid_loss = sess.run(full_loss, feed_dict={x: grid_data.valid_data, y_: grid_data.valid_labels})
      print("Validation loss: %f" % valid_loss)
      print("Train WMC: %f" % sess.run(tf.log(tf.reduce_mean(wmc)), feed_dict={x: grid_data.train_data[sup_train_inds, :], y_: grid_data.train_labels[sup_train_inds, :]}))
      if FLAGS.use_unlabeled:
        print("Unlabeled train WMC: %f" % sess.run(tf.log(tf.reduce_mean(wmc)), feed_dict={x: grid_data.train_data[unsup_train_inds, :], y_: grid_data.train_labels[unsup_train_inds, :]}))
      print("Validation WMC: %f" % sess.run(tf.reduce_mean(wmc), feed_dict={x: grid_data.valid_data, y_: grid_data.valid_labels}))

      print("Percentage of predictions that follow constraint: %f" % (float(np.sum([cmpe.weighted_model_count([(1-p, p) for p in np.concatenate((o, inp[24:]))]) for o, inp in zip(np.array(valid_out + 0.5, int), grid_data.valid_data)]))/float(grid_data.valid_data.shape[0])))
      
      # Early stopping
      if FLAGS.early_stopping:
        if prev_loss < valid_loss:
          print("Stopping early")
          test_out = sess.run(tf.nn.sigmoid(y), feed_dict={x: grid_data.test_data, y_: grid_data.test_labels})
          print("Percentage of test that are exactly right: %f" % (sum(1 for z in np.sum(np.abs(np.array(test_out + 0.5, int) - grid_data.test_labels), axis=1) if z == 0)/float(grid_data.test_labels.shape[0])))
          print("Percentage of individual labels in test that are right: %f" % (1. - np.sum(np.abs(np.array(test_out + 0.5, int) - grid_data.test_labels))/float(grid_data.test_labels.shape[0] * grid_data.test_labels.shape[1])))
          test_mpe_pred = np.array([cmpe.compute_mpe_inst([(1-p,p) for p in np.concatenate((o,inp[24:]))])[:24] for o, inp in zip(test_out, grid_data.test_data)])
          print("Test MPE accuracy %f" % (float(np.sum(np.equal(np.sum(np.abs(test_mpe_pred - grid_data.test_labels), axis = 1), 0))) / float(grid_data.test_labels.shape[0])))
          sys.exit()
        else:
          prev_loss = valid_loss

  # Test trained model
  correct_prediction = tf.equal(y, y_)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  # print(sess.run(W_1, feed_dict={x: mnist.test.images,
                                      # y_: mnist.test.labels}))


  train_out = sess.run(tf.nn.sigmoid(y), feed_dict={x: grid_data.train_data,
                                      y_: grid_data.train_labels})

  print(sess.run(accuracy, feed_dict={x: grid_data.valid_data,
                                      y_: grid_data.valid_labels}))
  print(sess.run(cross_entropy, feed_dict={x: grid_data.valid_data,
                                           y_: grid_data.valid_labels}))
  print(sess.run(tf.reduce_mean(wmc), feed_dict={x: grid_data.valid_data,
                                          y_: grid_data.valid_labels}))

  print(sum(1 for x in np.sum(np.abs(np.array(train_out + 0.5, int) - grid_data.train_labels), axis=1) if x == 0))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data', type=str, default='test.data',
                      help='Input data file to use')
  parser.add_argument('--units', type=int, default=100,
                      help='Number of units per hidden layer')
  parser.add_argument('--layers', type=int, default=3,
                      help='Number of hidden layers')
  parser.add_argument('--wmc', type=float, default=0.0,
                      help='Coefficient of WMC in loss')
  parser.add_argument('--iters', type=int, default=10000,
                      help='Number of minibatch steps to do')
  parser.add_argument('--relu', action='store_true',
                      help='Use relu hidden units instead of sigmoid')
  parser.add_argument('--early_stopping', action='store_true',
                      help='Enable early stopping - quit when validation loss is increasing')
  parser.add_argument('--give_labels', type=float, default=1.0,
                      help='Percentage of training examples to use labels for (1.0 = supervised)')
  parser.add_argument('--use_unlabeled', action='store_true',
                      help='Use this flag to enable semi supervised learning with WMC')
  parser.add_argument('--l2_decay', type=float, default=0.0,
                      help='L2 weight decay coefficient')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
