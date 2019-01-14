import tflearn
import speech_data
import tensorflow as tf

# 定义输入数据并预处理数据
learning_rate = 0.0001
training_iters = 300000     #迭代次数
batch_size = 64

width = 20  # MFCC特征
height = 80     # 最大发音长度
classes = 10    # 数字类别

# 对语言做分帧、取对数、逆矩阵等操作后，生成的MFCC 就代表这个语音的特征
batch = word_batch = speech_data.mfcc_batch_generator(batch_size)       # 生成每一批MFCC语音
X, Y = next(batch)
trainX, trainY = X, Y
testX, testY = X, Y

# 定义网络模型
net = tflearn.input_data([None, width, height])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')

# 训练模型
model = tflearn.DNN(net, tensorboard_verbose=0)
while 1:    # 训练迭代次数
    model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True, batch_size=batch_size)
    _y = model.predict(X)
model.save("tflearn.lstm.model")

# 预测模型
demo_file = "5_Vicki_260.wav"
demo = speech_data.load_wav_file(speech_data.path + demo_file)
result = mode.predict([demo])
result = numpy.argmax(result)
print("predicted digit for %s : result = %d "%(demo_file, result))


