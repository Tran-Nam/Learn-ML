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

graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./RNN/model-2000.meta')
        # saver.restore(sess, tf.train.latest_checkpoint('./RNN/'))
        saver.restore(sess, './RNN/model-2000')
        # print(sess.run('bias: 0'))
        # graph = tf.get_default_graph()
        # name = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        # print(name)
        X = graph.get_tensor_by_name('X: 0')
        Y = graph.get_tensor_by_name('Y: 0')
        # print(X.get_shape())
        # print(Y.get_shape())
        loss_op = graph.get_tensor_by_name('loss_op: 0')
        accuracy = graph.get_tensor_by_name('accuracy: 0')
        train_op = graph.get_operation_by_name('train_op')
        # train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_op)
        # print(X.get_shape())

        for step in range(1, training_step+1):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            # print(batch_x.shape)
            # print(batch_y.shape)

            sess.run(train_op, {X: batch_x, Y: batch_y})

            if step%display_step == 0 or step == 1:
                loss, acc = sess.run([loss_op, accuracy], {X:batch_x, Y: batch_y})

                print("Step " + str(step) + ", Minibatch Loss " + \
                "{:.4f}".format(loss)+ ", Training Accuracy: " + \
                "{:.3f}".format(acc))

        # save_path = saver.save(sess, './RNN/model_con')
        # print("Model saved in ", save_path)

        # print("Optimization Finished!")

        test_len = 128
        test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
        test_label = mnist.test.labels[:test_len]
        print("Testing Accuracy: ", \
        sess.run(accuracy, {X: test_data, Y: test_label}))