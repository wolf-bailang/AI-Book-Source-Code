import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 构建模型
# 定义输入数据并预处理数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
# 数据预处理把trX teX的形状变成[-1， 28， 28， 1], -1表示不考虑输入图片数量，1是通道数量
trX = trX.reshape(-1, 28, 28, 1)
teX = teX.reshape(-1, 28, 28, 1)

# 构建模型
# 设置超参数
lr = 0.001
training_iters = 100000
batch_size = 128

# 使用RNN 来分类图片，我们把每张图片的行看成是一个像素序列，28×28 像素，
# 所以我们把每一个图像样本看成一行行的序列，共有（28 个元素的序列）×（28 行），
# 然后每一步输入的序列长度是28，输入的步数是28 步
# 神经网络参数
n_inputs = 28    # 输入层的n
n_steps = 28    # 28长度
n_hidden_units = 128    # 隐藏岑的神经元个数
n_classes = 10      # 输出数量，即分类类别

# 定义输入数据及权重
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# 定义权重
weights = {# (28, 128)
           'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
           # (128, 10)
           'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
          }
biases = {# (128, )
          'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
          # (10, )
          'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
         }

# 定义RNN模型
def RNN(X, weights, biases):
    # 把输入的X转换成(128 batch * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # 进入隐藏层
    # X_in = (128 batch * 28 batch, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in = (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # 采用基本的LSTM: basic LSTM cell
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)

    # 初始化为0，lstm单元由两部分组成：（c_state, h_state)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    # dynamic_rnn接收张量（batch，steps，inputs）或者（steps，batch，inputs）作为X_in
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, intial_state=init_state, time_major=False)
    results = tf.matmul(final_state[1], weights['out']) + biases['out']

    return results

# 定义损失函数和优化器
pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

# 定义模型预测及准确率
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 训练数据和评估模型
# 启动会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps,n_inputs])
        sess.run([train_op], feed_dict={x: batch_xs, y: batch_ys})
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}))
        step += 1





