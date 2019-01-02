import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data # mnist手写数字识别库

###### 加载数据 ######

# 输入数据
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 定义网络参数
n_input = 784 # 输入的维度（img shape: 28*28）
n_classes = 10 # 标记的维度（0-9 digits）
dropout = 0.75 #Dropout的概率，输出的可能性

# 输入占位符
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) # dropout keep_prob留下的神经元的概率,如果keep_prob为0的话, 就是让所有的神经元都失活。

##### 构建网络模型 #####
# 定义卷积操作
def conv2d(name, x, W, b, strides=1):
    # strides在官方定义中是一个一维具有四个元素的张量，其规定前后必须为1，
    # 所以可以改的是中间两个数，中间两个数分别代表了水平滑动和垂直滑动步长值
    #‘SAME’表示填0操作，不足以扫描的元素位置抛弃掉，就是valid方式
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x, name=name) # 使用relu激活函数

# 定义池化层操作
def maxpool2d(name, x, k=2):
    # x: [batch, height, width, channels]
    # ksize: 池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，
    # 因为不想在batch和channels上做池化，所以这两个维度设为了1
    # stride: [1, stride,stride, 1]
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

# 规范化操作
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

##### 定义所有网络参数 #####
# 定义所有网络参数
weights ={'wc1': tf.Variable(tf.random_normal([11, 11, 1, 96])),    # [长/宽/深度/个数]
          'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
          'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
          'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
          'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
          'wd1': tf.Variable(tf.random_normal([4*4*256, 4096])),    # 全连接层6
          'wd2': tf.Variable(tf.random_normal(4096, 4096)),         # 全连接层7
          'out': tf.Variable(tf.random_normal([4096, 10]))          # 全连接层8
          }
biases = {'bc1': tf.Variable(tf.random_normal([96])),
          'bc2': tf.Variable(tf.random_normal([256])),
          'bc3': tf.Variable(tf.random_normal([384])),
          'bc4': tf.Variable(tf.random_normal([384])),
          'bc5': tf.Variable(tf.random_normal([256])),
          'bd1': tf.Variable(tf.random_normal([4096])),
          'bd2': tf.Variable(tf.random_normal([4096])),
          'out': tf.Variable(tf.random_normal([n_classes]))
          }

##### 定义AlexNet的网络模型 #####

# 定义整个网络
def alex_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # 第一卷积层
    # 卷积
    conv1 = conv2d('conv1', x, weights['wc1'], biases['bc1'])
    # 池化层
    pool1 = maxpool2d('pool1', conv1, k=2)
    # 规范化
    norm1 = norm('norm1', pool1, lsize=4)

    # 第二卷积层
    # 卷积
    conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
    # 池化层
    pool2 = maxpool2d('pool2', conv2, k=2)
    # 规范化
    norm2 = norm('norm2', pool2, lsize=4)

    # 第三卷积层
    # 卷积
    conv3 = conv2d('conv3', norm2, weights['wc3'], biases['bc3'])
    # 池化层
    pool3 = maxpool2d('pool3', conv3, k=2)
    # 规范化
    norm3 = norm('norm3', pool3, lsize=4)

    # 第四卷积层
    conv4 = conv2d('conv4', norm3, weights['wc4'], biases['bc4'])

    # 第五卷积层
    conv5 = conv2d('conv5', conv4, weights['wc5'], biases['bc5'])
    # 池化层
    pool5 = maxpool2d('pool5', conv5, k=2)
    # 规范化
    norm5 = norm('norm5', pool5, lsize=4)

    # 全连接层1
    fc1 = tf.reshape(norm5, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # 全连接层2
    fc2 = tf.reshape(fc1, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weights['wd1']), biases['bd1'])
    fc2 = tf.nn.relu(fc2)
    # dropout
    fc2 = tf.nn.dropout(fc2, dropout)

    # 输出层
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out

##### 构建模型，定义损失函数和优化器，并构建评估函数 #####

# 构建模型
pred = alex_net(x, weights, biases, keep_prob)

# 定义损失函数和优化器
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 评估函数
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

##### 训练和评估模型 #####

# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    # 开始训练，直到达到training_iters,即200000
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        if step % display_step == 0:
            # 计算损失值和准确度
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            print("Iter" + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f".format(acc))
        step += 1
    print("Optimization Finished !")

# 计算测试集的准确度
print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                                         y: mnist.test.labels[:256],
                                                         keep_prob: 1.}))
