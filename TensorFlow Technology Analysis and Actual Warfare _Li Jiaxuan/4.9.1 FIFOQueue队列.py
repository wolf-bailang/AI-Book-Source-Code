import tensorflow as tf

# 创建一个先入先出队列,初始化队列插入0.1、0.2、0.3 三个数字
q = tf.FIFOQueue(3, "float")
init = q.enqueue_many(([0.1, 0.2, 0.3],))

# 定义出队、+1、入队操作
x = q.dequeue()
y = x + 1
q_inc = q.enqueue([y])

# 开启会话，执行2次q_inc操作，查看队列内容
with tf.Session() as sess:
    sess.run(init)
    quelen = sess.run(q.size())
    for i in range(2):
        sess.run(q_inc) #执行2次操作，队列值变为0.3，1.1，1.2

    quelen = sess.run(q.size())
    for i in range(quelen):
        print(sess.run(q.dequeue())) #输出队列值