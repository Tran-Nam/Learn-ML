import numpy as np 
import tensorflow as tf 
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('./data', one_hot=True)


num_input = 28
timesteps = 28
num_hidden = 128
num_class = 10

X = tf.placeholder('float', [None, timesteps, num_input], name='X')
Y = tf.placeholder('float', [None, num_class], name='Y')

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./RNN/model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./RNN/'))
    # print(sess.run('bias: 0'))
    X = graph.get_tensor_by_name('X: 0')
