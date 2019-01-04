import numpy as np 
import tensorflow as tf 
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('./data', one_hot=True)


with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('./RNN/model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./RNN/'))
    print(sess.run('bias: 0'))