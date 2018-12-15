import tensorflow as tf

#创建一个含有队列的图
q = tf.FIFOQueue(1000, "float")
counter = tf.Variable(0.0) #计数器
increment_op = tf.assign_add(counter, tf.constant(1.0)) #操作：给计数器加1
enqueue_op = q.enqueue(counter) #操作：计数器值加入队列

#创建一个队列管理器QueueRunner
qr = tf.train.QueueRunner(q, enqueue_ops=[increment_op, enqueue_op] * 1)

