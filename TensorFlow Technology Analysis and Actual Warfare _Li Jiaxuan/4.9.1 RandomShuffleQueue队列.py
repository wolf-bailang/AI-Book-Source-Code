import tensorflow as tf

# 创建一个随机队列
q = tf.RandomShuffleQueue(capacity=10, min_after_dequeue=2, dtypes="float")

# 开启会话
with tf.Session() as sess:
    for i in range(0, 10): #10次入队
        sess.run(q.enqueue(i))

    for i in range(0, 8): #8次出队
        print(sess.run(q.dequeue())) 