import tensorflow as tf
import mnist

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

# 生成TFRecords文件
def main(unused_argv):
    #获取数据
    data_sets = mnist.read_data_sets(FLAGS.directory, dtype=tf.uint8, reshape=False, validation_size=FLAGS.validation_size)
    #将数据转换为tf.train.Example类型，并写入TFRecords文件
    convert_to(data_sets.train, 'train')
    convert_to(data_sets.validation, 'validation')
    convert_to(data_sets.test, 'test')

def convert_to(data_set, name):
    images = data_set.images
    labels = data_set.labels
    num_examples = data_set.num_examples # 55000个训练数据，5000个验证数据，10000个测试数据

    if images.shape[0] != num_examples:
        raise  ValueError('Images size %d does not match label size %d.' % (images.shape[0], num_examples))
    rows = images.shape[1] # 28
    cols = images.shape[2] # 28
    depth = images.shape[3] # 1,是黑白图像，所以是单通道

    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)

    for index in range(num_examples):
        image_raw = images[index].tostring()
        #写入协议缓冲区中，height, width, depth, label编码成int64类型，image_raw编码成二进制
        examaple = tf.train.Example(features=tf.train.Features(features={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(images_raw)}))
        writer.write(examaple.SerializeToString()) # 序列化为字符串
    writer.close()

# 编码函数
def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

# 从队列中读取
# 从文件中读取并解析一个样本
def read_and_decode(filename_queue): # 输入文件名队列
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(# 解析example
        serialized_example,
        # 必须写明features里面的key的名称 )
        features={
            'image_raw': tf.FixedLenFeature([], tf.string), # 图片是string类型
            'label': tf.FixedLenFeature([], tf.int64), #标记是int64类型
        })
    # 对于BytesList，要重新进行解码，把string类型的0维Tensor变成uint8类型的一维Tensor
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([mnist.IMAGEZ_PIXELS])
    # Tensor("input/DecodeRaw:0", shape=(784,), dtype=uint8)
    # image张量的形状为：Tensor("input/sub:0", shape=(784,), dtype=float32)
    image = tf.cast(image, tf.float32) * (1./255) - 0.5
    # 把标记从uint8类型转换为int32类型
    # label长老的形状为Tensor(input/Cast_1:0", shape=(), dtype=int32)
    label = tf.cast(features['label'], tf.int32)

    return image, label

# 用tf.train.shuffle_batch将样本随机化，获得一个最小批次的张量
def inputs(train, batch_size, num_epochs):
    # 输入参数
    # train: 选择输入训练数据/验证数据
    # batch_size: 训练的每一批有多少个样本
    # num_epochs: 过几遍数据，设置为0/None表示永远训练下去
    """
    返回结果：A tuple ( images, labels )
    images: 类型float, 形状[batch_size, mnist.IMAGE_PIXELS], 范围[-0.5, 0.5]
    labels: 类型int32, 形状[batch_size], 范围[0, mnist.NUM_CLASSES]
    注意tf.train.QueueRunner必须用tf.train.start_queue_runners()来启动线程
    """
    if not num_epochs: num_epochs = None
    # 获取文件路径，即/tmp/data/train,tfrecords, /tmp/data/validation.records
    filename = os.path.join(FLAGS.train_dir, TRAIN_FILE if train else VALIDATION_FILE)
    with tf.name_scope('input'):
        # tf.train.string_input_producer返回一个QueueRunner,里面有一个FIFOQueue
        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=num_epochs) # 如果样本量很大，可以分成若干文件，把文件名列表传入
        image, label = read_and_decode(filename_queue)
        # 随机化example, 并把它们规整成batch_size大小
        # tf.train.shuffle_batch生成了RandomShuffleQueue,并启动两个线程
        images, sparse_labels = tf.train.shuffle_batch(
            [image,label], batch_size=batch_size, num_threads=2,capacity=1000 + 3 * batch_size,
            min_after_dequeue=1000) # 留下一部分队列，来保证每次有足够的数据做随机打乱

        return images, sparse_labels

# 把batch张量作为输入，进行训练
def run_training():
    with tf.Graph().as_default():
        # 输入images和labels
        images, labels = inputs(train=True, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)
        # 构建一个从推理模型来预测数据的图
        logits = mnist.inference(images, FLAGS.hidden1, FLAGS.hidden2)
        loss = mnist.loss(logits, labels) # 定义损失函数

        # Add to the Graph operations that train the model.
        train_op = mnist.training(loss, FLAGS.learning_rate)

        # 初始化参数，string_input_producer内部创建了一个epoch计数变量，
        # 归入tf.GraphKeys.LOCAL_VARIABLES集合中，必须单独用initialize_local_variable()初始化
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess = tf.Session()
        sess.run(init_op)

        # Start input enqueue threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while not coord.should_stop(): # 进入死循环
                start_time = time.time()
                _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            # 每100次训练输出一次结果
            if step % 100 == 0 :
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
            step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
        finally:
            coord.request_stop() # 通知其他线程关闭

        coord.join(threads)
        sess.close()
