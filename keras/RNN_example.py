import numpy as np 
import tensorflow as tf 
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./data', one_hot=True)

lr = 1e-3
training_step = 10000
batch_size = 128
display_step = 200                                

num_input = 28
timesteps = 28
num_hidden = 128
num_class = 10

X = tf.placeholder('float', [None, timesteps, num_input])
Y = tf.placeholder('float', [None, num_class])

weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_class]))
}

bias = {
     
}