from keras.models import Sequential
from keras.layers import Dense, Activation

# 假设已加载完数据
# 构建模型
model = Sequential()
model.add(Dense(output_dim=64, input_dim=100))
model.add(Activation('relu'))
model.add(Dense(output_dim=10))
model.add(Activation('softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 训练和评估模型
model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)


# 在MNIST上用Keras实现一个CNN网络
# 定义超参数和加载数据
batch_size = 128
nb_classes = 10 # 分类数
nb_epoch = 12 # 训练次数

# 输入图片维度
img_rows, img_cols = 28, 28
# 卷积滤镜的个数
nb_fillters = 32
# 最大池化，池化核大小
pool_size = (2, 2)
# 卷积核大小
kernel_size = (3, 3)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

if K.image_dim_orderig() == 'th':
    # 使用Theano的顺序：（conv_dim1, channels, conv_dim2, conv_dim3)
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    # 使用TensorFlow的顺序：（conv_dim1, conv_dim2, conv_dim3, channels)
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# 将类向量转化为二进制类矩阵
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# 构建模型
model = Sequential()
model.add(Convolution2D(nb_fillters, kernel_size[0], kernel_size[1], border_model='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_fillters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add((Dropout(0.5)))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# 编译模型,采用多分类损失函数
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# 训练模型
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))

# 评估模型
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# 模型加载与保存
# 保存成.h5文件，包括模型结构、权重、训练配置（损失函数、优化器）
from keras.models import save_model, load_model

def test_sequential_model_saving():
    model = Sequential()
    model.add(Dense(2, input_dim=3))
    model.add(RepeatVector(3))
    model.add(TimeDistributed(Dense(3)))
    model.compile(loss=objectives.MSE, optimizer=optimizer.RMSprop(lr=0.001),
                  metrics=[metrics.categorical_accuracy],
                  sample_weight_mode='temporal')
    x = np.random.random((1, 3))
    y = np.random.random((1, 3, 3))
    model.train_on_batch(x, y)

    out = model.predict(x)
    _, fname = tempfile.mkstemp('.h5') # 创建一个HDFS 5 文件
    save_model(model, fname)

    new_model = load_model(fname)
    os.remove(fname)

    outs = new_model.predict(x)
    assert _allclose(out, out2, atol=1e-05)

    # 检测新保存的模型和之前定义的模型是否一致
    x = np.random.random((1, 3))
    y = np.random.random((1, 3, 3))
    model.train_on_batch(x, y)
    new_model.train_on_batch(x, y)
    out = model.predict(x)
    out2 = new_model.predict(x)
    assert _allclose(out, out2, atol=1e-05)

# json或yaml文件只保存模型结构，不包含权重和训练配置
json_string = model.to_json()
yaml_string = model.to_yaml()

# 加载
from  keras.models import model_from_json

model = model_from_json(json_string)
model = model_from_yaml(yaml_string)

# 用save_weights和load_weights只保存模型权重，不保存模型结构
model.save_weights('my_model_weights.h5')
model.load_werghts('my_model_weights.h5')

