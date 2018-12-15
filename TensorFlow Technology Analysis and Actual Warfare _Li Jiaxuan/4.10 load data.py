import tensorflow as tf

#预加载数据
x1 = tf.constant([2, 3, 4])
x2 = tf.constant([4, 0, 1])
y = tf.add(x1, x2)

#填充数据
#设计图
a1 = tf.placeholder(tf.int16)
a2 = tf.placeholder(tf.int16)
b = tf.add(a1, a2)
#用python产生数据
li1 = [2, 3, 4]
li2 = [4, 0, 1]
#打开一个会话，将数据填充给后端
with tf.Session() as sess:
    print(sess.run(b, feed_dict={a1: li1, a2: li2}))




