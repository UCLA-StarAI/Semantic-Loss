from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

import numpy

from mnist_input import read_data_sets

import tensorflow as tf

FLAGS = None

def standarization(x_images):
  with tf.name_scope('standarization'):
    x_images = tf.reshape(x_images, [-1, 28, 28, 1])
    x_images  = tf.map_fn(lambda image: tf.image.per_image_standardization(image), x_images)
    x_images = tf.reshape(x_images, [-1, 784])
  return x_images

def image_processing(x_images):
  with tf.name_scope('gaussian_noise'):
    noise = tf.random_normal(tf.shape(x_images), mean = 0.0, stddev = 0.3, dtype=tf.float32)
    x_images = x_images + noise
  with tf.name_scope('crop'):
    x_images = tf.reshape(x_images, [-1,28,28,1])
    x_images = tf.random_crop(x_images, [FLAGS.batch_size, 25, 25, 1])
    x_images = tf.image.resize_image_with_crop_or_pad(x_images, 28, 28)
    x_images = tf.reshape(x_images, [-1,784])
  return x_images

def deepnn(x_images):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  keep_prob = tf.placeholder(tf.float32)

  x_images = standarization(x_images)
  x_images = tf.cond(keep_prob < 1.0, lambda: image_processing(x_images), lambda: x_images)
 
  with tf.name_scope('fc1'):
    h_fc1 = tf.contrib.layers.fully_connected(x_images, 1000)
    
  with tf.name_scope('fc2'):
    h_fc2 = tf.contrib.layers.fully_connected(h_fc1, 500)
    
  with tf.name_scope('fc3'):
    h_fc3 = tf.contrib.layers.fully_connected(h_fc2, 250)
    
  with tf.name_scope('fc4'):
    h_fc4 = tf.contrib.layers.fully_connected(h_fc3, 250)

  with tf.name_scope('fc5'):
    h_fc5 = tf.layers.batch_normalization(tf.contrib.layers.fully_connected(h_fc4, 250))
  
  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    h_fc5_drop = tf.nn.dropout(h_fc5, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc6'):
    y_mlp = tf.contrib.layers.fully_connected(h_fc5, 10, activation_fn = None)

  return y_mlp, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def leakyRelu(value, alpha=0.01):
  return tf.maximum(value, alpha*value)

def main(_):
  # Import data
  mnist = read_data_sets(FLAGS.data_path, n_labeled=FLAGS.num_labeled, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Build the graph for the deep net
  y_mlp, keep_prob = deepnn(x)

  batch_number = tf.shape(y_mlp)[0]
  label_examples = tf.greater(tf.reduce_max(y_, axis=1), tf.zeros([batch_number,]))
  label_examples = tf.cast(label_examples, tf.float32)

  with tf.name_scope('cross_entropy'):
    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_mlp)
    cross_entropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y_mlp), axis=1)
    
  with tf.name_scope('wmc'):
    normalized_logits = tf.nn.sigmoid(y_mlp)
    wmc_tmp = tf.zeros([batch_number,])
    for i in range(10):
        one_situation = tf.concat(
          [tf.concat([tf.ones([batch_number, i]), tf.zeros([batch_number, 1])], axis=1),
           tf.ones([batch_number, 10-i-1])], axis=1)
        wmc_tmp += tf.reduce_prod(one_situation - normalized_logits, axis=1)
  wmc_tmp = tf.abs(wmc_tmp)
  wmc = tf.reduce_mean(wmc_tmp)

  with tf.name_scope('loss'):
    unlabel_examples = tf.ones([batch_number,]) - label_examples
    log_wmc = tf.log(wmc_tmp)
    loss = -0.0005*tf.multiply(unlabel_examples, log_wmc) - 0.0005*tf.multiply(label_examples, log_wmc) + tf.multiply(label_examples, cross_entropy)
    loss = tf.reduce_mean(loss)
  
  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_mlp, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    correct_prediction = tf.multiply(correct_prediction, label_examples)
  accuracy = tf.reduce_sum(correct_prediction) / tf.reduce_sum(label_examples)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_average_accuracy, train_average_wmc, train_average_loss = 0.0, 0.0, 0.0
    for i in range(50000):
      images, labels = mnist.train.next_batch(FLAGS.batch_size)
      _, train_accuracy, train_wmc, train_loss =  sess.run([train_step, accuracy, wmc, loss], feed_dict={x: images, y_: labels, keep_prob: 0.5})
      train_average_accuracy += train_accuracy
      train_average_wmc += train_wmc
      train_average_loss += train_loss

      if i % 100 == 0:
        train_average_accuracy /= 100
        train_average_wmc /= 100
        train_average_loss /= 100
        with open("log.txt", 'a') as outFile:
          print('step %d, training_accuracy %g, train_loss %g, wmc %g' % (i, train_average_accuracy, train_average_loss, train_average_wmc))
          outFile.write('step %d, training_accuracy %g, train_loss %g, wmc %g\n' % (i, train_average_accuracy, train_average_loss, train_average_wmc))
          train_average_accuracy, train_average_wmc, train_average_loss = 0.0, 0.0, 0.0
      if i % 500 == 0:

          test_accuracy = accuracy.eval(feed_dict={
              x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
          with open("log.txt", 'a') as outFile:
            print('test accuracy %g' % (test_accuracy))
            outFile.write('test accuracy %g\n' % (test_accuracy))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', type=str,
                      default='mnist_data/',
                      help='Directory for storing input data')
  parser.add_argument('--num_labeled', type=int,
                      help='Num of labeled examples provided for semi-supervised learning.')
  parser.add_argument('--batch_size', type=int,
                      help='Batch size for mini-batch Adams gradient descent.')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
