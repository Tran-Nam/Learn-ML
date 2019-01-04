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

X = tf.placeholder('float', [None, timesteps, num_input], name='X')
Y = tf.placeholder('float', [None, num_class], name='Y')

weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_class]), name='weights')
}

bias = {
     'out': tf.Variable(tf.random_normal([num_class]), name='bias')
}

def RNN(x, weights, bias):
    x = tf.unstack(x, timesteps, 1)
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + bias['out']

logits = RNN(X, weights, bias)
prediction = tf.nn.softmax(logits, name='prediction')

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for step in range(1, training_step+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))

        sess.run(train_op, {X: batch_x, Y: batch_y})

        if step%display_step == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], {X:batch_x, Y: batch_y})

            print("Step " + str(step) + ", Minibatch Loss " + \
            "{:.4f}".format(loss)+ ", Training Accuracy: " + \
            "{:.3f}".format(acc))
        
    save_path = saver.save(sess, './RNN/model')
    print("Model saved in ", save_path)

    # print("Optimization Finished!")

    # test_len = 128
    # test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    # test_label = mnist.test.labels[:test_len]
    # print("Testing Accuracy: ", \
    # sess.run(accuracy, {X: test_data, Y: test_label}))


# saver = tf.train.Saver()
# with tf.Session() as sess:
# with tf.Session() as sess:
#     saver.restore(sess, './RNN/model.ckpt')
#     print("Model restored!")

#     test_len = 128
#     test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
#     test_label = mnist.test.labels[:test_len]
#     print("Testing Accuracy: ", \
#     sess.run(accuracy, {X: test_data, Y: test_label}))

# with tf.Session() as sess:
#     saver.restore(sess, './RNN/model.ckpt')
#     print("Model restored!")

#     test_len = 128
#     test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
#     test_label = mnist.test.labels[:test_len]
#     print("Testing Accuracy: ", \
#     sess.run(accuracy, {X: test_data, Y: test_label}))

