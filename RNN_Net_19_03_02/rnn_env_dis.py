import tensorflow as tf
import os
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from Data_Processing import dataRebuild
import random
# 定义一个生成数据的 get_batch function:
def get_batch(trainData):
    global BATCH_START, TIME_STEPS,INITSTATE
    BATCH_START = random.randint(0, len(trainData) - 1 - 2 * TIME_STEPS * BATCH_SIZE)
    xs = trainData[BATCH_START:BATCH_START + TIME_STEPS*BATCH_SIZE, :].reshape((-1,TIME_STEPS ,INPUT_SIZE))
    ys = trainData[BATCH_START + 1:BATCH_START + TIME_STEPS*BATCH_SIZE + 1, [0, 1, 2]].reshape((-1,TIME_STEPS,OUTPUT_SIZE))
    if BATCH_START == (int(len(trainData)/BATCH_SIZE) - 2) * BATCH_SIZE:
        INITSTATE = True
        BATCH_START = 0
    return [xs, ys, xs]

# 定义 LSTMRNN 的主体结构
class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, learning_rate, norm = False):
        self.norm = norm
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.lr = learning_rate
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
            self.Txs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='Txs')
            self.test = tf.placeholder(tf.float32, [None, n_steps, output_size], name='test')
            self.Vxs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='Vxs')
            self.val = tf.placeholder(tf.float32, [None, n_steps, output_size],name='val')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
            tf.add_to_collection('train_op', self.train_op)
        with tf.name_scope('VT_process'):
            self.computeTestCost()
            self.computeValcost()

    def normalize(self, data_in, dim, dim_size):
        fc_mean, fc_var = tf.nn.moments(
            data_in,
            axes=[dim]
        )
        scale = tf.Variable(tf.ones([dim_size]))
        shift = tf.Variable(tf.zeros([dim_size]))
        epsilon = 0.001
        # apply moving average for mean and var when train on batch
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([fc_mean, fc_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(fc_mean), tf.identity(fc_var)

        mean, var = mean_var_with_update()

        output = tf.nn.batch_normalization(data_in, mean, var, shift, scale, epsilon)
        return output

    # 设置 add_input_layer 功能, 添加 input_layer:
    def add_input_layer(self, ):
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
        # Ws (in_size, cell_size)
        with tf.name_scope('input_W'):
            Ws_in = self._weight_variable([self.input_size, self.cell_size])
            tf.summary.histogram('input_W', Ws_in)
        # bs (cell_size, )
        with tf.name_scope('input_B'):
            bs_in = self._bias_variable([self.cell_size, ])
            tf.summary.histogram('input_B', bs_in)
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Xin_input_W_plus_input_B'):
            l_in_y = tf.nn.relu(tf.matmul(l_in_x, Ws_in) + bs_in)
            tf.summary.histogram('input_layer_y',l_in_y)
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')
        if self.norm    :
            self.l_in_y = self.normalize(self.l_in_y, 0, self.cell_size)

    # 设置 add_cell 功能, 添加 cell, 注意这里的 self.cell_init_state,
    #  因为我们在 training 的时候, 这个地方要特别说明.
    def add_cell(self):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)
        if self.norm :
            self.cell_outputs = self.normalize(self.cell_outputs, 0, self.cell_size)
        tf.summary.histogram('rnn_out', self.cell_outputs)

    # 设置 add_output_layer 功能, 添加 output_layer:
    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        with tf.name_scope('output_W'):
            Ws_out = self._weight_variable([self.cell_size, self.output_size])
            tf.summary.histogram('output_W', Ws_out)
        with tf.name_scope('output_B'):
            bs_out = self._bias_variable([self.output_size, ])
            tf.summary.histogram('output_B', bs_out)
        # shape = (batch * steps, output_size)
        with tf.name_scope('Xout_output_W_plus_output_B'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out
            tf.summary.histogram('prediction', self.pred)
            tf.add_to_collection('pred', self.pred)

    # RNN 误差计算部分:
    def compute_cost(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.n_steps * self.output_size], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
                name='average_cost')
            tf.summary.scalar('cost', self.cost)
            tf.add_to_collection('average_cost', self.cost)

    def computeTestCost(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.test, [-1], name='reshape_testTarget')],
            [tf.ones([self.batch_size * self.n_steps * self.output_size], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='testLosses'
        )
        with tf.name_scope('average_testCost'):
            self.testCost = tf.div(
                tf.reduce_sum(losses, name='testLosses_sum'),
                self.batch_size,
                name='average_testCost')
            tf.summary.scalar('testCost', self.testCost)
            tf.add_to_collection('average_testCost', self.testCost)

    def computeValcost(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.val, [-1], name='reshape_valTarget')],
            [tf.ones([self.batch_size * self.n_steps * self.output_size], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='valLosses'
        )
        with tf.name_scope('average_valCost'):
            self.valCost = tf.div(
                tf.reduce_sum(losses, name='valLosses_sum'),
                self.batch_size,
                name='average_valCost')
            tf.summary.scalar('valCost', self.valCost)
            tf.add_to_collection('average_valCost', self.valCost)

    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1., )
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)

def get_VTData(data):
    batch_start = random.randint(0, len(data) - 1 - TIME_STEPS*BATCH_SIZE)
    VTxin = data[batch_start:batch_start + TIME_STEPS*BATCH_SIZE , :].reshape((-1,TIME_STEPS ,INPUT_SIZE))
    VTlabel = data[batch_start + 1: batch_start + TIME_STEPS*BATCH_SIZE + 1, [0, 1, 2]].reshape((-1,TIME_STEPS,OUTPUT_SIZE))
    return [VTxin, VTlabel]

# 训练 LSTMRNN
if __name__ == '__main__':
    BATCH_START = 0  # 建立 batch data 时候的 index
    TIME_STEPS = 1  # backpropagation through time 的time_steps 误差反传的步长
    BATCH_SIZE = 1000 #每次训练的步长的数量
    INPUT_SIZE = 6  # 数据输入size
    OUTPUT_SIZE = 3  # 数据输出 size
    CELL_SIZE = 512  # RNN的 hidden unit size
    LR = 0.006  # learning rate
    INITSTATE = False  # 判断是否读到数据的末尾，通过置零BATCHB_START开启第二轮训练
    MODEL_NAME = 'iteration_model-1822000'
    # 加载数据
    buildData = dataRebuild('Data.txt', 0.7, 0.15)
    trainData = np.loadtxt('trainData.txt')
    testData = np.loadtxt('testData.txt')
    valData = np.loadtxt('valData.txt')
    # 搭建 LSTMRNN 模型
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, LR, norm = True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=0.5)
    if os.path.exists('model') and os.path.exists('model/%s.meta'%MODEL_NAME) :
        print('restore last model')
        saver.restore(sess, 'model/%s'%MODEL_NAME)
    else:
        print('create a new model')
        sess.run(tf.global_variables_initializer())


    # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
    # $ tensorboard --logdir='logs'

    # matplotlib可视化
    plt.ion()  # 设置连续 plot
    plt.show()
    fig = plt.figure(1, figsize=(24,12), dpi=80)
    iteration = 0
    cost_list = []
    trainLoss_list = []
    testLoss_list = []
    valLoss_list = []
    # 训练
    while(True):
        iteration += 1
        seq, res, xs = get_batch(trainData)  # 提取 batch data
        #xs = np.arange(0,50)
        if iteration == 1 or INITSTATE:
            INITSTATE = False
            # 初始化 data
            feed_dict = {
                model.xs: seq,
                model.ys: res,
                model.Txs: get_VTData(testData)[0],
                model.test: get_VTData(testData)[1],
                model.Vxs: get_VTData(valData)[0],
                model.val: get_VTData(valData)[1]
            }
        else:
            feed_dict = {
                model.xs: seq,
                model.ys: res,
                model.Txs: get_VTData(testData)[0],
                model.test: get_VTData(testData)[1],
                model.Vxs: get_VTData(valData)[0],
                model.val: get_VTData(valData)[1],
                model.cell_init_state: state  # 保持 state 的连续性
            }

        # 训练
        _, cost, state, pred = sess.run(
            [model.train_op, model.cost, model.cell_final_state, model.pred],
            feed_dict=feed_dict)
        cost_list.append(cost)

        # 每50次迭代记录一次log 并print cost
        if iteration%50 == 0 :
            if len(trainLoss_list) == 250:
                valLoss_list = []
                trainLoss_list = []
                # testLoss_list = []
            valLoss = sess.run(model.valCost, feed_dict=feed_dict)
            # cost保留四位小数输出
            print('iteration: ', iteration, 'train_cost: ', round(cost, 4), 'val_cost:', round(valLoss, 4), 'test_cost:', 'None')
            # # # 测试验证模型
            # testLoss_list.append(testLoss)
            valLoss_list.append(valLoss)
            trainLoss_list.append(np.mean(cost_list))
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, iteration)


        # 每1000次迭代保存一次模型 并打印cost曲线
        if iteration % 2500 == 0:

            # # # 保存模型
            saver.save(sess, 'model/iteration_model', global_step=iteration)

            # # # 绘制数据动态图
            label = res.reshape(-1, OUTPUT_SIZE)
            output = pred.reshape(-1, OUTPUT_SIZE)
            plt.clf()
            chart_1 = fig.add_subplot(7, 1, 1, xlim=(0, label.shape[0]))
            chart_1.set_title('S_L')
            chart_2 = fig.add_subplot(7, 1, 2, xlim=(0, label.shape[0]))
            chart_2.set_title('S_M')
            chart_3 = fig.add_subplot(7, 1, 3, xlim=(0, label.shape[0]))
            chart_3.set_title('S_R')
            # chart_4 = fig.add_subplot(7, 1, 4, xlim=(0, label.shape[0]))
            # chart_4.set_title('Dis_1')
            # chart_5 = fig.add_subplot(7, 1, 5, xlim=(0, label.shape[0]))
            # chart_5.set_title('Dis_2')
            chart_6 = fig.add_subplot(7, 1, 6, xlim=(0, len(cost_list)))
            chart_6.set_title('trainLoss')
            chart_7 = fig.add_subplot(7, 1, 7, xlim=(0, len(trainLoss_list)))
            chart_7.set_title('train_test_val_loss')

            # S_L
            chart_1.scatter(np.arange(0, label.shape[0]), label[:, 0], c='blue', s=10, marker='o')
            chart_1.scatter(np.arange(0, label.shape[0]), output[:, 0], c='red', s=10, marker='x')
            # S_M
            chart_2.scatter(np.arange(0, label.shape[0]), label[:, 1], c='blue', s=10, marker='o')
            chart_2.scatter(np.arange(0, label.shape[0]), output[:, 1], c='red', s=10, marker='x')
            # S_R
            chart_3.scatter(np.arange(0, label.shape[0]), label[:, 2], c='blue', s=10, marker='o')
            chart_3.scatter(np.arange(0, label.shape[0]), output[:, 2], c='red', s=10, marker='x')
            # # Dis_1
            # chart_4.plot(np.arange(0, label.shape[0]), label[:, 3], 'b-', lw=4)
            # chart_4.plot(np.arange(0, label.shape[0]), output[:, 3], 'r--', lw=2)
            # # Dis_2
            # chart_5.plot(np.arange(0, label.shape[0]), label[:, 4], 'b-', lw=4)
            # chart_5.plot(np.arange(0, label.shape[0]), output[:, 4], 'r--', lw=2)
            # Cost
            chart_6.plot(np.arange(0, len(cost_list)), cost_list, 'r-', lw = 2)
            # VT_cost
            # chart_7.plot (np.arange(0, len(testLoss_list)), testLoss_list, 'g-', lw = 2)
            chart_7.plot(np.arange(0, len(valLoss_list)), valLoss_list, 'r--', lw = 2)
            chart_7.plot(np.arange(0, len(trainLoss_list)), trainLoss_list, 'b--', lw = 2)
            plt.pause(0.1)
            if not os.path.exists('plot'):
                os.makedirs('plot')
            plt.savefig('plot/iteration_%s.png' % iteration)
            cost_list = []


