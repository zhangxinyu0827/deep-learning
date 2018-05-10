import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import matplotlib.pyplot as plt

BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 16
LR = 0.006

def get_batch():
    global BATCH_START, TIME_STEPS
    x = np.arange(BATCH_START, BATCH_START + BATCH_SIZE * TIME_STEPS).reshape(
        (BATCH_SIZE, TIME_STEPS)) / (10 * np.pi)

    sinx = np.sin(x)
    cosx = np.cos(x)

    BATCH_START += TIME_STEPS
    return sinx[:, :, np.newaxis], cosx[:, :, np.newaxis], x

class RNN():
    def __init__(self, x):
        cell = rnn.BasicLSTMCell(CELL_SIZE)
        self.cell_init_state = cell.zero_state(BATCH_SIZE, dtype=tf.float32)

        outputs, self.final_state = tf.nn.dynamic_rnn(
            cell, x, initial_state=self.cell_init_state, time_major=False)
        outputs = tf.reshape(outputs, (-1, CELL_SIZE))
        w = tf.Variable(tf.random_normal([CELL_SIZE, OUTPUT_SIZE]))
        b = tf.Variable(tf.random_normal([OUTPUT_SIZE]))
        self.pred = tf.matmul(outputs, w) + b

def ms_error(labels, logits):
    return tf.square(tf.subtract(labels, logits))

X = tf.placeholder(tf.float32, [None, TIME_STEPS, INPUT_SIZE])
Y = tf.placeholder(tf.float32, [None, TIME_STEPS, OUTPUT_SIZE])
rnn = RNN(X)
pred = rnn.pred

cost = tf.reduce_mean(
    tf.square(tf.subtract(tf.reshape(pred, [-1]), tf.reshape(Y, [-1]))))
optimizer = tf.train.AdamOptimizer(LR).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
plt.figure(figsize=(10, 5))
plt.ion()
plt.show()
for i in range(200):
    sinx, cosx, x = get_batch()
    if i == 0:
        feed_dict = {
            X: sinx,
            Y: cosx,
        }
    else:
        feed_dict = {
            X: sinx,
            Y: cosx,
            rnn.cell_init_state: state,
        }
    _, state, outputs = sess.run(
        [optimizer, rnn.final_state, pred], feed_dict=feed_dict)

    plt.plot(x[0, :], cosx[0].flatten(), 'r', x[0, :],
             outputs.flatten()[:TIME_STEPS], 'b--')
    plt.ylim((-1.2, 1.2))
    plt.xlim(x[0, 0] - 10, x[0, -1])
    plt.draw()
    plt.pause(0.1)

    if i % 20 == 0:
        print('cost: ', sess.run(cost, feed_dict=feed_dict))