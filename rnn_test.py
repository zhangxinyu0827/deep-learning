
#activate tensorflow
import tensorflow as tf
from tensorflow.contrib import rnn

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/", one_hot=True)

lr = 0.001#学习率
training_iters = 100000
batch_size = 32#一次扔进去多少张图进行训练
display_step = 10

n_input = 28
n_step = 28
n_hidden = 64  #层数决定了每个LSTM结构输出的H的维数，对于最后一个h，h*w+b既是我们的预测结果，中间的过程我们不管
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_step, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
w = tf.Variable(tf.random_normal([n_hidden, n_classes]))
b = tf.Variable(tf.random_normal([n_classes]))

def RNN(x):
    # x (batch, n_step, n_input)
    # x = tf.unstack(x, n_step, 1)
    cell = rnn.BasicLSTMCell(n_hidden)

    # static_rnn
    # outputs: (n_step, batch, n_hidden)
    # state[0]: (batch, n_hidden), state is a tuple
    # outputs, state = rnn.static_rnn(cell, x, dtype=tf.float32)
    # -------------------------------------------------------
    # outputs: (batch, n_step, n_hidden)
    outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    return tf.matmul(tf.transpose(outputs, [1, 0, 2])[-1], w) + b

pred = RNN(x)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('loss', cost)
summaries = tf.summary.merge_all()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('logs/', sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 1
    while batch_size * step < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape(batch_size, n_step, n_input)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            acc, loss = sess.run(
                [accuracy, cost], feed_dict={x: batch_x,y: batch_y})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        if step % 100 == 0:
            s = sess.run(summaries, feed_dict={x: batch_x, y: batch_y})
            train_writer.add_summary(s, global_step=step)

        step += 1
    print("Optimization Finished!")

    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_step, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={x: test_data, y: test_label}))