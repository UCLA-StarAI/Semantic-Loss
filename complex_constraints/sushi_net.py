import argparse
import sys

import numpy as np

from sushi_data import SushiData, to_pairwise_comp
from compute_mpe import CircuitMPE

import tensorflow as tf

FLAGS =None

def weight_variable(shape):
  return tf.Variable(tf.truncated_normal(shape, 0.1))

def bias_variable(shape):
  return tf.Variable(tf.truncated_normal(shape, 0.1))

def main(_):
    # Get data
    sushi_data = SushiData('sushi.soc')
    INPUT_SIZE = sushi_data.train_data.shape[1]
    OUTPUT_SIZE = sushi_data.train_labels.shape[1]
    # Create the model
    # Input(25) - Layer 1(units)
    x = tf.placeholder(tf.float32, [None, INPUT_SIZE])
    W = []
    b = []
    ys = []
    W.append(weight_variable([INPUT_SIZE, FLAGS.units]))
    b.append(bias_variable([FLAGS.units]))
    ys.append(tf.nn.sigmoid(tf.matmul(x, W[0]) + b[0]))

    for i in range(1, FLAGS.layers):
        # Layer i(units) - Layer i+1(units)
        W.append(weight_variable([FLAGS.units, FLAGS.units]))
        b.append(bias_variable([FLAGS.units]))
        ys.append(tf.nn.sigmoid(tf.matmul(ys[i-1], W[i]) + b[i]))

    # Layer n(units) - Output(25)
    W.append(weight_variable([FLAGS.units, OUTPUT_SIZE]))
    b.append(bias_variable([OUTPUT_SIZE]))
    y = tf.matmul(ys[-1], W[-1]) + b[-1] + np.finfo(float).eps * 10


    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])
    yu = tf.unstack(tf.nn.sigmoid(y), axis=1)

    # Create AC
    # yleaves = [[ny, 1.0 - ny] for ny in yu]
    # ac = AC('permutation-4.ac',yleaves)

    # Create CircuitMPE instance for our predictions
    cmpe = CircuitMPE('permutation-4.vtree', 'permutation-4.sdd')
    wmc = cmpe.get_tf_ac([[1.0 - ny,ny] for ny in yu])

    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))
    loss = cross_entropy - FLAGS.wmc * tf.log(tf.reduce_mean(wmc))
    # train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    # train_step = tf.train.AdagradOptimizer(0.1).minimize(loss)
    # train_step = tf.train.MomentumOptimizer(0.1, 0.5).minimize(loss)
    train_step = tf.train.AdamOptimizer().minimize(loss)

    # Train loop
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Should only compute pairwise comparisons once for labels
    train_pw = to_pairwise_comp(sushi_data.train_labels, np.sqrt(OUTPUT_SIZE))
    valid_pw = to_pairwise_comp(sushi_data.valid_labels, np.sqrt(OUTPUT_SIZE))
    test_pw = to_pairwise_comp(sushi_data.test_labels, np.sqrt(OUTPUT_SIZE))

    print train_pw.shape
    print train_pw

    # For early stopping
    prev_loss = 1e15
    # Train
    for i in range(FLAGS.iters):
        # batch_xs, batch_ys = sushi_data.get_batch(400)
        batch_xs, batch_ys = sushi_data.train_data, sushi_data.train_labels
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        # Every 1k iterations check accuracy
        if i % 100 == 0:
            print("After %d iterations" % i)

            # Computing "true" accuracies
            correct_prediction = tf.equal(tf.reduce_sum(tf.abs(tf.to_int32(tf.nn.sigmoid(y)+0.5) - tf.to_int32(y_)), 1), 0)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print "Train accuracy: %f" % sess.run(accuracy, feed_dict={x: sushi_data.train_data, y_: sushi_data.train_labels})
            print "Validation accuracy: %f" % sess.run(accuracy, feed_dict={x: sushi_data.valid_data, y_: sushi_data.valid_labels})

            # Computing MPE instiation accuracies
            pred = sess.run(tf.nn.sigmoid(y), feed_dict={x: sushi_data.train_data, y_: sushi_data.train_labels})
            mpe_pred = np.array([cmpe.compute_mpe_inst([(1-p, p) for p in o]) for o in pred])
            print "Train mpe accuracy: %f" % (float(np.sum(np.equal(np.sum(np.abs(mpe_pred - sushi_data.train_labels), axis=1), 0))) / float(sushi_data.train_data.shape[0]))
            valid_pred = sess.run(tf.nn.sigmoid(y), feed_dict={x: sushi_data.valid_data, y_: sushi_data.valid_labels})
            valid_mpe_pred = np.array([cmpe.compute_mpe_inst([(1-p, p) for p in o]) for o in valid_pred])
            print "Validation mpe accuracy: %f" % (float(np.sum(np.equal(np.sum(np.abs(valid_mpe_pred - sushi_data.valid_labels), axis=1), 0))) / float(sushi_data.valid_data.shape[0]))

            # Percentage of individual labels that are right
            print("Percentage of individual labels in training that are right: %f" % (1. - np.sum(np.abs(np.array(pred + 0.5, int) - sushi_data.train_labels))/float(sushi_data.train_labels.shape[0] * sushi_data.train_labels.shape[1])))
            print("Percentage of individual labels in validation that are right: %f" % (1. - np.sum(np.abs(np.array(valid_pred + 0.5, int) - sushi_data.valid_labels))/float(sushi_data.valid_labels.shape[0] * sushi_data.valid_labels.shape[1])))


            # Compute pairwise accuracies using MPE
            mpe_pred_pw = to_pairwise_comp(mpe_pred, 5)
            print "Train pairwise mpe accuracy: %f" % (1. - float(np.sum(np.abs(mpe_pred_pw - train_pw)))/float(10*sushi_data.train_data.shape[0]))
            valid_mpe_pred_pw = to_pairwise_comp(valid_mpe_pred, 5)
            print "Validation pairwise mpe accuracy: %f" % (1. - float(np.sum(np.abs(valid_mpe_pred_pw - valid_pw)))/float(10*sushi_data.valid_data.shape[0]))

            # Print loss
            print "Train loss: %f" % sess.run(loss, feed_dict={x: sushi_data.train_data, y_: sushi_data.train_labels})
            valid_loss = sess.run(loss, feed_dict={x: sushi_data.valid_data, y_: sushi_data.valid_labels})
            print "Validation loss: %f" % valid_loss

            # Print WMC
            print "Train WMC: %f" % sess.run(tf.reduce_mean(wmc), feed_dict={x: sushi_data.train_data, y_: sushi_data.train_labels})
            print "Validation WMC: %f" % sess.run(tf.reduce_mean(wmc), feed_dict={x: sushi_data.valid_data, y_: sushi_data.valid_labels})

            print("Percentage of predictions that follow constraint: %f" % (float(np.sum([cmpe.weighted_model_count([(1-p, p) for p in o]) for o in np.array(valid_pred + 0.5, int)]))/float(sushi_data.valid_data.shape[0])))

            # Early stopping
            if prev_loss < valid_loss:
                print "Stopping early"
                print "Test accuracy: %f" % sess.run(accuracy, feed_dict={x: sushi_data.test_data, y_: sushi_data.test_labels})
                test_pred = sess.run(tf.nn.sigmoid(y), feed_dict={x: sushi_data.test_data, y_: sushi_data.test_labels})
                test_mpe_pred = np.array([cmpe.compute_mpe_inst([(1-p, p) for p in o]) for o in test_pred])
                print "Test mpe accuracy: %f" % (float(np.sum(np.equal(np.sum(np.abs(test_mpe_pred - sushi_data.test_labels), axis=1), 0))) / float(sushi_data.test_data.shape[0]))
                print("Percentage of individual labels in test that are right: %f" % (1. - np.sum(np.abs(np.array(test_pred + 0.5, int) - sushi_data.test_labels))/float(sushi_data.test_labels.shape[0] * sushi_data.test_labels.shape[1])))
                test_mpe_pred_pw = to_pairwise_comp(test_mpe_pred, 5)
                print "Test pairwise mpe accuracy: %f" % (1. - float(np.sum(np.abs(test_mpe_pred_pw - test_pw)))/float(10*sushi_data.test_data.shape[0]))

                sys.exit()
            else:
                prev_loss = valid_loss




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Args go here
    parser.add_argument('--units', type=int, default=100,
                      help='Number of units per hidden layer')
    parser.add_argument('--layers', type=int, default=3,
                        help='Number of hidden layers')
    parser.add_argument('--wmc', type=float, default=0.0,
                        help='Coefficient of WMC in loss')
    parser.add_argument('--iters', type=int, default=10000,
                        help='Number of minibatch steps to do')


    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
