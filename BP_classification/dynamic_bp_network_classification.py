import tensorflow as tf
import pandas as pd
import numpy as np
createVar = locals()


'''
建立一个网络结构可变的BP神经网络通用代码：

在训练时各个参数的意义：
num_class: 分类数
hidden_floors_num：隐藏层的个数
every_hidden_floor_num：每层隐藏层的神经元个数
learning_rate：学习速率
activation：激活函数
regularization：正则化方式
regularization_rate：正则化比率
total_step：总的训练次数
train_data_path：训练数据路径，假如有3类，用0,1,2进行标记
validate_data_path：验证数据路径
model_save_path：模型保存路径

利用训练好的模型进行预测时各个参数的意义：
model_save_path：模型的保存路径
predict_data_path：预测数据路径
predict_result_save_path：预测结果保存路径
'''

# 训练模型全局参数
num_class = 2
hidden_floors_num = 2
every_hidden_floor_num = [4, 3]
learning_rate = 0.03
activation = 'tanh'
regularization = 'L1'
regularization_rate = 0.0001
total_step = 1000000
train_data_path = 'C:/Users/Administrator/Desktop/BP_classification/train.csv'
validate_data_path = 'C:/Users/Administrator/Desktop/BP_classification/validate.csv'
model_save_path = 'C:/Users/Administrator/Desktop/BP_classification/model/classification_model'


# 利用模型进行预测全局参数
model_save_path = 'C:/Users/Administrator/Desktop/BP_classification/model/classification_model'
predict_data_path = 'C:/Users/Administrator/Desktop/BP_classification/test.csv'
predict_result_save_path = 'C:/Users/Administrator/Desktop/BP_classification/test_predict.csv'


def inputs(train_data_path):
    train_data = pd.read_csv(train_data_path)
    X = np.array(train_data.iloc[:, :-1])
    Y = np.array(train_data.iloc[:, -1:])
    label = np.zeros([X.shape[0], num_class])
    for i in range(X.shape[0]):
        label[i, int(Y[i])] = 1
    return X, label


def make_hidden_layer(pre_lay_num, cur_lay_num, floor):
    createVar['w' + str(floor)] = tf.Variable(tf.random_normal([pre_lay_num, cur_lay_num], stddev=1))
    createVar['b' + str(floor)] = tf.Variable(tf.random_normal([cur_lay_num], stddev=1))
    return eval('w'+str(floor)), eval('b'+str(floor))


def initial_w_and_b(all_floors_num):
    # 初始化隐藏层的w, b
    for floor in range(2, hidden_floors_num+3):
        pre_lay_num = all_floors_num[floor-2]
        cur_lay_num = all_floors_num[floor-1]
        w_floor, b_floor = make_hidden_layer(pre_lay_num, cur_lay_num, floor)
        createVar['w' + str(floor)] = w_floor
        createVar['b' + str(floor)] = b_floor


def cal_floor_output(x, floor):
    w_floor = eval('w'+str(floor))
    b_floor = eval('b'+str(floor))
    if activation == 'sigmoid':
        output = tf.sigmoid(tf.matmul(x, w_floor) + b_floor)
    if activation == 'tanh':
        output = tf.tanh(tf.matmul(x, w_floor) + b_floor)
    if activation == 'relu':
        output = tf.nn.relu(tf.matmul(x, w_floor) + b_floor)
    return output


def inference(x):
    output = x
    for floor in range(2, hidden_floors_num+2):
        output = cal_floor_output(output, floor)

    floor = hidden_floors_num+2
    w_floor = eval('w'+str(floor))
    b_floor = eval('b'+str(floor))
    output = tf.matmul(output, w_floor) + b_floor
    return output


def loss(x, y_real):
    y_pre = inference(x)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pre, labels=tf.argmax(y_real, 1))
    if regularization == 'None':
        total_loss = tf.reduce_mean(cross_entropy)

    if regularization == 'L1':
        total_loss = 0
        for floor in range(2, hidden_floors_num + 3):
            w_floor = eval('w' + str(floor))
            total_loss = total_loss + tf.contrib.layers.l1_regularizer(regularization_rate)(w_floor)
        total_loss = total_loss + tf.reduce_mean(cross_entropy)

    if regularization == 'L2':
        total_loss = 0
        for floor in range(2, hidden_floors_num + 3):
            w_floor = eval('w' + str(floor))
            total_loss = total_loss + tf.contrib.layers.l2_regularizer(regularization_rate)(w_floor)
        total_loss = total_loss + tf.reduce_mean(cross_entropy)

    return total_loss


def train(total_loss):
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
    return train_op


# 训练模型
def train_model(num_class, hidden_floors_num, every_hidden_floor_num, learning_rate, activation, regularization,
                regularization_rate, total_step, train_data_path, validate_data_path, model_save_path):
    X, Y = inputs(train_data_path)
    X_validation , Y_validation = inputs(validate_data_path)
    X_dim = X.shape[1]
    all_floors_num = [X_dim] + every_hidden_floor_num + [num_class]

    # 将参数保存到和model_save_path相同的文件夹下， 恢复模型进行预测时加载这些参数创建神经网络
    temp = model_save_path.split('/')
    model_name = temp[-1]
    parameter_path = ''
    for i in range(len(temp)-1):
        parameter_path = parameter_path + temp[i] + '/'
    parameter_path = parameter_path + model_name + '_parameter.txt'
    with open(parameter_path, 'w') as f:
        f.write("all_floors_num:")
        for i in all_floors_num:
            f.write(str(i) + ' ')
        f.write('\n')
        f.write('activation:')
        f.write(str(activation))

    x = tf.placeholder(dtype=tf.float32, shape=[None, X_dim])
    y_real = tf.placeholder(dtype=tf.float32, shape=[None, num_class])
    initial_w_and_b(all_floors_num)
    y_pre = inference(x)
    total_loss = loss(x, y_real)
    train_op = train(total_loss)

    # 记录在训练集上的正确率
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y_real, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 保存模型
    saver = tf.train.Saver()

    # 在一个会话对象中启动数据流图，搭建流程
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    for step in range(total_step):
        sess.run([train_op], feed_dict={x: X[:, :], y_real: Y[:, :]})
        if step % 1000 == 0:
            saver.save(sess, model_save_path)
            total_loss_value = sess.run(total_loss, feed_dict={x: X[:, :], y_real: Y[:, :]})
            print('train step is ', step, ', total loss value is ',
                  total_loss_value, 'train accuracy is ', sess.run(accuracy, feed_dict={x: X[:, :], y_real: Y[:, :]}),
                  'validation accuracy is ', sess.run(accuracy,
                                                      feed_dict={x: X_validation[:, :], y_real: Y_validation[:, :]}))

    saver.save(sess, model_save_path)
    sess.close()


# 利用模型进行类别预测
def predict(model_save_path, predict_data_path, predict_result_save_path):
    # **********************根据model_save_path推出模型参数路径, 解析出all_floors_num和activation****************
    temp = model_save_path.split('/')
    model_name = temp[-1]
    parameter_path = ''
    for i in range(len(temp)-1):
        parameter_path = parameter_path + temp[i] + '/'
    parameter_path = parameter_path + model_name + '_parameter.txt'
    with open(parameter_path, 'r') as f:
        lines = f.readlines()

    # 从读取的内容中解析all_floors_num
    temp = lines[0].split(':')[-1].split(' ')
    all_floors_num = []
    for i in range(len(temp)-1):
        all_floors_num = all_floors_num + [int(temp[i])]

    # 从读取的内容中解析activation
    activation = lines[1].split(':')[-1]
    hidden_floors_num = len(all_floors_num) - 2

    # **********************读取预测数据*************************************
    predict_data = pd.read_csv(predict_data_path)
    X = np.array(predict_data.iloc[:, :])
    X_dim = X.shape[1]

    # **********************创建神经网络************************************
    x = tf.placeholder(dtype=tf.float32, shape=[None, X_dim])
    initial_w_and_b(all_floors_num)
    y_pre = inference(x)

    sess = tf.Session()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 读取模型
        try:
            saver.restore(sess, model_save_path)
            print('模型载入成功！')
        except:
            print('模型不存在，请先训练模型！')
            return
        y_pre_value = sess.run(y_pre, feed_dict={x: X[0:, :]})
        y_pre_value = np.argmax(y_pre_value, 1)
        # 将预测结果写入csv文件
        predict_data_columns = list(predict_data.columns) + ['label']
        data = np.column_stack([X, y_pre_value])
        result = pd.DataFrame(data, columns=predict_data_columns)
        result.to_csv(predict_result_save_path, index=False)
        print('预测结果保存在：', predict_result_save_path)


if __name__ == '__main__':
    mode = 'predict'

    if mode == 'train':
        # 训练模型
        train_model(num_class, hidden_floors_num, every_hidden_floor_num, learning_rate, activation, regularization,
                    regularization_rate, total_step, train_data_path, validate_data_path, model_save_path)

    if mode == 'predict':
        # 利用模型进行预测
        predict(model_save_path, predict_data_path, predict_result_save_path)












