from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 加载数据
# FLAGS.data_dir 是MNIST 所在的路径，用户可以自己指定
# 使用one_hot 的直接原因是，我们使用0～9 个类别的多分类的输出层是softmax 层，它的输
# 出是一个概率分布，从而要求输入的标记也以概率分布的形式出现，进而可以计算交叉熵
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

# 构建回归模型
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b     # 预测值

# 定义损失函数和优化器
y_ = tf.placeholder(tf.float32, [None, 10])     #输入的真实值占位符

# 用tf.nn.softmax_cross_entropy_with_logits来计算预测值y和真实值y_的差值，并取均值
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
# 采用SGD优化器
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 训练模型
# 用InteractiveSession()来创建交互式上下文的TensorFlow会话
# 方法（如tf.Tensor.eval和tf.Operation.run)都可以使用该会话进行运行操作
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train
for _ in range(1000)
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

# 评估模型
# tf.argmax(vector, 1):返回的是vector中的最大值的索引号
# tf.argmax(y,1)返回的是模型对任一输入x预测到的标记值，tf.argmax(y_,1)代表正确的标记值
# tf.equal 来检测预测值和真实值是否匹配，并且将预测后得到的布尔值转化成浮点数，并取平均值
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))    # 计算预测值和真实值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 布尔型转化为浮点数，取均值得准确率
# 计算测试集准确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
