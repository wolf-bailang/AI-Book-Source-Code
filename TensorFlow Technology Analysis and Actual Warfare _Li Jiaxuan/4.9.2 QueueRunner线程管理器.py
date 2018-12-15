import tensorflow as tf

#创建一个含有队列的图
q = tf.FIFOQueue(1000, "float")
counter = tf.Variable(0.0) #计数器
increment_op = tf.assign_add(counter, tf.constant(1.0)) #操作：给计数器加1
enqueue_op = q.enqueue(counter) #操作：计数器值加入队列

#创建一个队列管理器QueueRunner
qr = tf.train.QueueRunner(q, enqueue_ops=[increment_op, enqueue_op] * 1)

#启动一个会话，从队列管理器qr 中创建线程
#主线程
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    enqueue_threads = qr.create_threads(sess, start=True) # 启动入队线程
    #主线程
    for i in range(10):
    print (sess.run(q.dequeue()))

#使用协调器（coordinator）来管理线程
#主线程
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Coordinator: 协调器，协调线程间的关系可以视为一种信号量，用来做同步
coord = tf.train.Coordinator()

#启动入队线程，协调器是线程的参数
enqueue_threads = qr.create_threads(sess, coord = coord, start = True)

coord.request_stop() #通知其他线程关闭
#主线程
for i in range(0, 10):
    #使用tf.errors.OutOfRangeError 来捕捉错误，终止循环
    try:
        print(sess.run(q.dequeue()))
    except tf.errors.OutOfRangeError:
        break

coord.join(enqueue_threads) #join操作等待其他线程结束，其他所有线程关闭之后，这一函数才能返回



