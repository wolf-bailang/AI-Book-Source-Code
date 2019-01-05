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

X = tf.placeholder("float", [None, 28, 28 ,1])
Y = tf.placeholder("float", [None, 10])

# 初始化权重和定义网络结构
def init_weights(shape):
    # tf.random_normal用于从服从指定正太分布的数值中取出指定个数的值
    # mean: 正态分布的均值，默认为0
    # stddev: 正态分布的标准差，默认为1.0
    return tf.Variable(tf.random_normal(shape, stddev=0.01))
# 初始化权重，卷积核大小3*3
w = init_weights([3, 3, 1, 32])     # 输入维度1，输出维度32
w2 = init_weights([3, 3, 32, 64])   # 输入维度32，输出维度64
w3 = init_weights([3, 3, 64, 128])   # 输入维度64，输出维度128
w4 = init_weights([128 * 4 * 4, 625])   # 全连接层，输入维度128*4*4是上一层的输出数据三维转一维，输出维度128
w_o = init_weights([625, 10])       # 输出层，输入维度为 625, 输出维度为10，代表10 类(labels)

# 定义一个模型
# 神经网络模型的构建函数，传入参数
# X：输入数据
# w: 每一层的权重
# p_keep_conv, p_keep_hidden: dropout 要保留的神经元比例
def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    # 第一组卷积层及池化层，最后dropout一些神经元
    l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'))
    # l1a shape=(?, 28, 28, 32)
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # l1 shape=(?, 14, 14, 32)
    l1 = tf.nn.dropout(l1, p_keep_conv)

    # 第二组卷积层及池化层，最后dropout一些神经元
    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'))
    # l2a shape=(?, 14, 14, 64)
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # l2 shape=(?, 7, 7, 64)
    l2 = tf.nn.dropout(l2, p_keep_conv)

    # 第三组卷积层和池化层，最后dropout一些神经元
    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME'))
    # l3a shape=(? , 4, 4, 128)
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # l3 shape=(?, 4, 4, 128)
    l3 = tf.reshape(13, [-1, w4.get_shape().as_list()[0]])      # reshape to (?, 2048)
    l3 = tf.nn.dropout(13, p_keep_conv)

    # 全连接层，最后dropout一些神经元
    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    # 输出层
    pyx = tf.matmul(l4, w_o)
    return pyx      # 返回预测值

# 定义dropout占位符，生成网络模型，得到预测值
p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)     # 得到预测值

# 定义损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)     # 学习率为0.001，衰减值为0.9
predict_op = tf.argmax(py_x, 1)

# 训练和评估模型
# 定义训练批次大小和评估批次大小
batch_size = 128
test_size = 256
# 开启会话
with tf.Session() as sess:
    # 初始化全局变量
    tf.global_variables_initializer().run()

    for i in range(100):
        # zip()将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象
        training_batch = zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end], p_keep_conv: 0.8, p_keep_hidden: 0.5}))
        # np.arange([start, ]stop, [step, ]dtype=None)用于创建等差数组
        # start:可忽略不写，默认从0开始;起始值
        # stop:结束值；生成的元素不包括结束值
        # step:可忽略不写，默认步长为1；步长
        # dtype:默认为None，设置显示元素的数据类型
        test_indices = np.arange(len(teX))      # 得到测试批次大小
        # np.random.shuffle(x) 现场修改序列，改变自身内容。（类似洗牌，打乱顺序）
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]
        print(i, np.mean(np.argmax(teY[test_indices], axis=1) == sess.run(predict_op, feed_dict={X: teX[test_indices], p_keep_conv: 1.0, p_keep_hidden: 1.0})))



