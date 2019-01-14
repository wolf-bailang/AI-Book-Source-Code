
# 数据集 康奈尔大学Corpus数据集
# 数据集整理成问和答文件，生成.enc（问句）和.dec（答句）文件
# test.dec # 测试集答句
# test.enc # 测试集问句
# train.dec # 训练集答句
# train.enc # 训练集问句

# 创建词汇表,然后把问句和答句转换成对应的id形
# vocab20000.dec # 答句的词汇表
# vocab20000.enc # 问句的词汇表
# 转换成的ids 文件如下：
# 问句和答句转换成的ids 文件中，每一行是一个问句或答句，每一行中的每一个id 代表问句或答句中对应位置的词
# test.enc.ids20000
# train.dec.ids20000
# train.enc.ids20000

# 定义训练参数
# 将参数写到一个专门的文件seq2seq.ini 中
[strings]
# 模式：train, test, serve
mode = train
train_enc = data/train.enc
train_dec = data/train.dec
test_enc = data/test.enc
test_dec = data/test.dec
# 模型文件和词汇表的存储路径
working_directory = working_dir/
[ints]
# 词汇表大小
enc_vocab_size = 20000
dec_vocab_size = 20000
# LSTM层数
num_layers = 3
# 每层大小
layer_size = 256

max_train_data_size = 0
batch_size = 64
# 每多少次迭代存储一次模型
steps_per_checkpoint = 300
[floats]
learning_rate = 0.5 # 学习率
learning_rate_decay_factor = 0.99 # 学习速率下降系数
max_gradient_norm = 5.0

# 定义网络模型
# seq2seq + Attention模型
# 该模型的代码在seq2seq_model.py 中
class Seq2SeqModel(object):

    def __init__(self, source_vocab_size, target_vocab_size, buckets, size, num_layers, max_gradient_norm,
                 batch_size, learning_rate, learning_rate_decay_factor, use_lstm=False, num_samples=512,
                 forward_only=False):
        """构建模型
        参数：
        source_vocab_size: 问句词汇表大小
        target_vocab_size: 答句词汇表大小
        buckets: (I,O), 其中I 的定最大输入长度，O 的定最大输出长度
        size: 每一层的神经元数量
        num_layers: 模型层数
        max_gradient_norm: 梯度将被削减到最大的规范
        batch_size: 批次大小。用于训练和预测的批次大小，可以不同
        learning_rate: 学习速率
        learning_rate_decay_factor: 调整学习速率
        use_lstm: 使用LSTM 的元来代替GRU 的元
        num_samples: 使用softmox 的样本数
        forward_only: 是否仅构建前向传播
        """
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        output_projection = None
        softmax_loss_function = None
        # 如果样本量比词汇表的量小，那么要用抽样的softmax
        if num_samples > 0 and num_samples < self.target_vocab_size:
            w = tf.get_variable("proj_w", [size, self.target_vocab_size]) #tf.get_variable获取一个已经存在的变量或者创建一个新的变量
            w_t = tf.transpose(w)     # 转置
            b = tf.get_variable("proj_b", [size, self.target_vocab_size])
            output_projection = (w, b)

            def sampled_loss(inputs, labels):
                labels = tf.reshape(labels, [-1, -1])
                return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples, self.target_vocab_size)
                softmax_loss_function = sampled_loss

            # 构建RNN
            single_cell = tf.nn.rnn_cell.GRUCell(size)
            if use_lstm:
                single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
            cell = single_cell
            if num_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

            # Attention模型
            def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
                return tf.nn.seq2seq.embedding_attention_seq2seq(encoder_inputs, decoder_inputs, cell,
                                                                 num_encoder_symbols=source_vocab_size,
                                                                 num_decoder_symbols=target_vocab_size,
                                                                 embedding_size=size,
                                                                 output_projection=output_projection,
                                                                 feed_previous=do_decode)
            # 给模型填充数据
            self.encoder_inputs = []
            self.decoder_inputs = []
            self.targer_weights = []
            for i in xrange(buckets[-1][0]):
                self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                          name="encoder{0}".format(i)))
            for i in xrange(buckets[-1][1] + 1):
                self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                          name="decoder{0}".format(i)))
                self.targer_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                          name="weight{0}".format(i)))
            # targets的值是解码器偏移1位
            targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]

            # 训练模型的输出
            if forward_only:
                self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(self.encoder_inputs,
                                                                             self.decoder_inputs,
                                                                             targets,
                                                                             self.targer_weights,
                                                                             buckets,
                                                                             lambda x, y: seq2seq_f(x, y, True),
                                                                             softmax_loss_function=softmax_loss_function)
                if output_projection is not None:
                    for b in xrange(len(buckets)):
                        self.outputs[b] = [tf.matmul(output, output_projection[0]) + output_projection[1]
                                           for output in self.outputs[b]]
            else:
                self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(self.encoder_inputs,
                                                                             self.decoder_inputs,
                                                                             targets,
                                                                             self.targer_weights,
                                                                             buckets,
                                                                             lambda x, y: seq2seq_f(x, y, False),
                                                                             softmax_loss_function=softmax_loss_function)
            # 训练模型时，更新梯度
            params = tf.trainable_variables()  # 返回的是需要训练的变量列表
            if not forward_only:
                self.gradient_norms = []
                self.updates = []
                opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                for b in xrange(len(buckets)):
                    gradients = tf.gradients(self.losses[b], params)  # 计算梯度的函数
                    # tf.clip_by_global_norm通过权重梯度的总和的比率来截取多个张量的值。
                    clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                    self.gradient_norms.append(norm)
                    self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))
                self.saver = tf.train.Saver(tf.all_variables()) # 保存参数

            # 定义运行模型每一步
            def step(self, session, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only):
                """运行模型每一步
                参数：
                    session: tensorflow session
                    encoder_inputs: 问句向量序列
                    decoder_inputs: 答句向量序列
                    target_weights: target weights
                    bucket_id: 输入的bucket_id
                    forward_only: 是否只做前向传播
                """
                encoder_size, decoder_size = self.buckets[bucket_id]
                if len(encoder_inputs) != encoder_size:
                    raise ValueError("Encoder length must be equal to the one in bucket,"
                                     " %d != %d." % (len(encoder_inputs), encoder_size))
                if len(decoder_inputs) != decoder_size:
                    raise ValueError("Decoder length must be equal to the one in bucket,"
                                     " %d != %d." % (len(decoder_inputs), decoder_size))
                if len(target_weights) != decoder_size:
                    raise ValueError("Weights length must be equal to the one in bucket,"
                                     " %d != %d." % (len(target_weights), decoder_size))

                # 输入填充
                input_feed = {}
                for l in xrange(encoder_size):
                    input_feed[self.encoder_inputs[1].name] = encoder_inputs[1]
                for l in xrange(decoder_size):
                    input_feed[self.decoder_inputs[1].name] = decoder_inputs[1]
                    input_feed[self.target_weights[1].name] = target_weights[1]

                last_target = self.decoder_inputs[decoder_size].name
                input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

                # 输出填充，与是否后向传播有关
                if not forward_only:
                    output_feed = [self.updates[bucket_id],
                                   self.gradient_norms[bucket_id],
                                   self.losses[bucket_id]]
                else:
                    output_feed = [self.losses[bucket_id]]
                    for l in xrange(decoder_size):
                        output_feed.append(self.outputs[bucket_id][1])
                outputs = session.run(output_feed, input_feed)
                if not forward_only:
                    return outputs[1], outputs[2], None     # 有后向传播下的输出：梯度，损失值，None
                else:
                    return None, outputs[0], outputs[1:]    # 仅有前向传播的输出：None，损失值，outouts

            # 为训练的每一步产生一个批次的数据
            def get_batch(self, data, bucket_id):
                """
                这个函数的作用时从指定的桶中获取一个批次的随机数据，在训练的每一步中使用
                :param data: 长度为self.buckets的元组，其中每个元素都包含用于创建批次的输入和输出数据的列表
                :param bucket_id: 整数，从哪个bucket获取本批次
                :return: 一个包含三项的元组（encoder_inputs, decoder_inputs, target_weights)
                """

# 训练模型
# 修改seq2seq.ini文件中的mode值，当值为train时，可以运行execute.py进行训练
def train():
    # 准备数据集
    print("Preparing data in %s" % gConfig['working_directory'])
    enc_train, dec_train, enc_dev, dec_dev, _, _ = data_utils.prepare_custom_data(gConfig['working_directory'],
                                                                                  gConfig['train_enc'],
                                                                                  gConfig['train_dec'],
                                                                                  gConfig['test_enc'],
                                                                                  gConfig['test_dec'],
                                                                                  gConfig['enc_vocab_size'],
                                                                                  gConfig['dec_vocab_size'])
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    with tf.Session(config=config) as sess:
        # 构建模型
        print("Creating %d layers of %d units." % (gConfig['num_layers'], gConfig['layer_size']))
        model = create_model(sess, False)
        # 把数据读入桶（bucket）中，并计算桶的大小
        print("Reading development and training data (limit: %d)." % gConfig['max_train_data_size'])
        dec_set = read_data(enc_dev, dec_dev)
        train_set = read_data(enc_train, dec_train,gConfig['max_train_data_size'])
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_bucket_scale = [sum(train_bucket_sizes[:i + 1]) / train_bucket_sizes for i in xrange(len(train_bucket_sizes))]

        # 开始训练循环
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            # 随机生成一个0-1的数字，在生成bucket_id中使用
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_bucket_scale)) if train_bucket_scale[i] > random_number_01])
            # 获取一个批次的数据，并进一步训练
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / gConfig['steps_per_checkpoint']
            loss += step_loss / gConfig['steps_per_checkpoint']
            current_step += 1

            # 保存检查点文件，打印统计数据
            if current_step % gConfig['steps_per_checkpoint'] == 0:
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print("global step %d learning rate %.4f step-time %.2f perplexity" "%.2f" % (model.global_step.eval(),
                                                                                              model.learning_rate.eval(),
                                                                                              step_time, perplexity))
                # 如果损失值在最近3次内没有再降低，就减小学习率
                if len(previous_losses) > 2 and loss >max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # 保存检查点文件，并把计数器和损失值归零
                checkpoint_path = os.path.join(gConfig['working_directory'], "seq2seq.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0

# 验证模型
# 修改seq2seq.ini文件中mode值，当为test时，可以运行execute.py进行测试
def decode():
    with tf.Session() as sess:
        # 建立模型，并定义超参数batch_size
        model = creat_model(sess, True)
        # 加载词汇表文件
        enc_vocab_path = os.path.join(gConfig['working_directory'], "vocab%d.enc" % gConfig['enc_vocab_size'])
        dec_vocab_path = os.path.join(gConfig['working_directory'], "vocab%d.dec" % gConfig['dec_vocab_size'])
        enc_vocab, _ = data_utils.initialize_vocabulary(enc_vocab_path)
        _, rev_dec_vocab = data_utils.initialize_vocabulary(dec_vocab_path)

        # 对标准输入的句子进行解码
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdout.readline()
        while sentence:
            # 得到输入句子的token-ids
            token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), enc_vocab)
            # 计算这个token_ids属于哪一个桶bucket
            bucket_id = min([b for b in xrange(len(_buckets)) if  _buckets[b][0] > len(token_ids)])
            # 将句子送入到模型中
            encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(token_ids, [])]},
                                                                             bucket_id)
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
            # 这是一个贪心的解码器，输出只是output_logits的argmaxes
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            # 如果输出中有EOS符号，在EOS处切断
            if data_utils.EOS_ID in outputs:
                outputs = outputs[:outputs.index(data_utils.EOS_ID)]
            # 打印出与输出句子对应的法语句子
            print(" ".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs]))
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()




