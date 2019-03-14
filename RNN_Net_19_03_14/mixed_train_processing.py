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
    xs = trainData[BATCH_START:BATCH_START + TIME_STEPS*BATCH_SIZE,  :].reshape((-1,TIME_STEPS ,INPUT_SIZE))
    ys = trainData[BATCH_START + 1:BATCH_START + TIME_STEPS*BATCH_SIZE + 1, :-1].reshape((-1,TIME_STEPS,OUTPUT_SIZE))
    if BATCH_START == (int(len(trainData)/BATCH_SIZE) - 2) * BATCH_SIZE:
        INITSTATE = True
        BATCH_START = 0
    return [xs, ys, xs]

# 定义 LSTMRNN 的主体结构
class LSTMRNN_D(object):
    def __init__(self, n_steps, input_size, output_size, D_input_size, D_output_size, S_input_size, S_output_size,
                 cell_size, batch_size, learning_rate, norm = False):
        self.norm = norm
        self.n_steps = n_steps
        self.D_input_size = D_input_size
        self.D_output_size = D_output_size
        self.S_input_size = S_input_size
        self.S_output_size = S_output_size
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.lr = learning_rate
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
            # print(self.ys[:,:, 3:5])
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.D_train_op = tf.train.AdamOptimizer(self.lr).minimize(self.D_cost)
            self.S_train_op = tf.train.AdamOptimizer(self.lr).minimize(self.S_cost)
            tf.add_to_collection('D_train_op', self.D_train_op)
            tf.add_to_collection('S_train_op', self.S_train_op)

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
        with tf.name_scope('Dis_input_layer'):
            # D_Ws (D_in_size, cell_size)
            with tf.name_scope('D_input_W'):
                D_Ws_in = self._weight_variable([self.D_input_size, self.cell_size], name = 'D_input_W')
                tf.summary.histogram('D_input_W', D_Ws_in)
            # D_bs (cell_size, )
            with tf.name_scope('D_input_B'):
                D_bs_in = self._bias_variable([self.cell_size, ], name='D_input_B')
                tf.summary.histogram('D_input_B', D_bs_in)
            # D_l_in_y = (batch * n_steps, cell_size)
            with tf.name_scope('D_Xin_input_W_plus_input_B'):
                D_l_in_y = tf.matmul(l_in_x[:, 3:], D_Ws_in) + D_bs_in
                tf.summary.histogram('input_layer_y',D_l_in_y)
        # reshape D_l_in_y ==> (batch, n_steps, cell_size)
        self.D_l_in_y = tf.reshape(D_l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

        with tf.name_scope('Sensor_input_layer'):
            # S_Ws (S_in_size, cell_size)
            with tf.name_scope('S_input_W'):
                S_Ws_in = self._weight_variable([self.S_input_size, self.cell_size], name='S_input_W')
                tf.summary.histogram('S_input_W', S_Ws_in)
            # S_bs(cell_size,)
            with tf.name_scope('S_input_B'):
                S_bs_in = self._bias_variable([self.cell_size,],name='S_input_B')
                tf.summary.histogram('S_input_B', S_bs_in)
            # S_l_in_y = (batch * n_steps, cell_size)
            with tf.name_scope('S_Xin_input_X_plus_input_B'):
                S_l_in_y = tf.matmul(l_in_x, S_Ws_in) + S_bs_in
        # reshape S_l_in_y ==> (batch, n_steps, cell_size)
        self.S_l_in_y = tf.reshape(S_l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

        #normalize
        if self.norm    :
            with tf.name_scope('normalize'):
                self.D_l_in_y = self.normalize(self.D_l_in_y, 0, self.cell_size)
                self.S_l_in_y = self.normalize(self.S_l_in_y, 0, self.cell_size)

    # 设置 add_cell 功能, 添加 cell, 注意这里的 self.cell_init_state,
    #  因为我们在 training 的时候, 这个地方要特别说明.
    def add_cell(self):
        with tf.name_scope('D_RNN_cell'):
            D_lstm_cell = tf.nn.rnn_cell.LSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True, name='D_RNN_cell')
            with tf.name_scope('D_initial_state'):
                self.D_cell_init_state = D_lstm_cell.zero_state(self.batch_size, dtype=tf.float32)

            self.D_cell_outputs, self.D_cell_final_state = tf.nn.dynamic_rnn(
                D_lstm_cell, self.D_l_in_y, initial_state=self.D_cell_init_state, time_major=False, )
            if self.norm :
                self.D_cell_outputs = self.normalize(self.D_cell_outputs, 0, self.cell_size)
            tf.summary.histogram('rnn_out', self.D_cell_outputs)

        with tf.name_scope('S_RNN_cell'):
            S_lstm_cell = tf.nn.rnn_cell.LSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True, name='S_RNN_cell')
            with tf.name_scope('S_initial_state'):
                self.S_cell_init_state = S_lstm_cell.zero_state(self.batch_size, dtype=tf.float32)

            self.S_cell_outputs, self.S_cell_final_state = tf.nn.dynamic_rnn(
                S_lstm_cell, self.S_l_in_y, initial_state=self.S_cell_init_state, time_major=False, )
            if self.norm:
                self.S_cell_outputs = self.normalize(self.S_cell_outputs, 0, self.cell_size)
            tf.summary.histogram('rnn_out', self.S_cell_outputs)

    # 设置 add_output_layer 功能, 添加 output_layer:
    def add_output_layer(self):
        with tf.name_scope('D_output_layer'):
            # shape = (batch * steps, cell_size)
            D_l_out_x = tf.reshape(self.D_cell_outputs, [-1, self.cell_size], name='2_2D')
            with tf.name_scope('D_output_W'):
                D_Ws_out = self._weight_variable([self.cell_size, self.D_output_size], 'D_output_W')
                tf.summary.histogram('D_output_W', D_Ws_out)
            with tf.name_scope('D_output_B'):
                D_bs_out = self._bias_variable([self.D_output_size, ],'D_output_B')
                tf.summary.histogram('D_output_B', D_bs_out)
            # shape = (batch * steps, D_output_size)
            with tf.name_scope('D_Xout_output_W_plus_output_B'):
                self.pred_D = tf.matmul(D_l_out_x,D_Ws_out) + D_bs_out
                tf.summary.histogram('prediction_D', self.pred_D)
                tf.add_to_collection('pred_D', self.pred_D)

        with tf.name_scope('S_output_layer'):
            S_l_out_x = tf.reshape(self.S_cell_outputs, [-1, self.cell_size], name='2_2D')
            with tf.name_scope('S_output_W'):
                S_Ws_out = self._weight_variable([self.cell_size, self.S_output_size], name='S_ouput_W')
                tf.summary.histogram('S_output_W', S_Ws_out)
            with tf.name_scope('S_output_B'):
                S_bs_out = self._bias_variable([self.S_output_size, ], name='S_output_B')
                tf.summary.histogram('S_output_B', S_bs_out)
            # shape = (batch * steps, S_output_size)
            with tf.name_scope('S_Xout_output_W_plus_output_B'):
                self.pred_S = tf.matmul(S_l_out_x, S_Ws_out) + S_bs_out
                tf.summary.histogram('prediction_S', self.pred_S)
                tf.add_to_collection('pred_S', self.pred_S)

    # RNN 误差计算部分:
    def compute_cost(self):

        D_losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred_D, [-1], name='reshape_pred_D')],

            [tf.reshape(self.ys[:,:, 3:5], [-1], name='reshape_target_D')],
            [tf.ones([self.batch_size * self.n_steps * self.D_output_size], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='D_losses'
        )
        with tf.name_scope('D_average_cost'):
            self.D_cost = tf.div(
                tf.reduce_sum(D_losses, name='D_losses_sum'),
                self.batch_size,
                name='D_average_cost')
            tf.summary.scalar('D_cost', self.D_cost)
            tf.add_to_collection('D_average_cost', self.D_cost)

        S_losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred_S, [-1], name='reshape_pred_S')],
            [tf.reshape(self.ys[:,:, :3], [-1], name='reshape_target_S')],
            [tf.ones([self.batch_size * self.n_steps * self.S_output_size], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='S_losses'
        )
        with tf.name_scope('S_average_cost'):
            self.S_cost = tf.div(
                tf.reduce_sum(S_losses, name='S_losses_sum'),
                self.batch_size,
                name='S_average_cost')
            tf.summary.scalar('S_cost', self.S_cost)
            tf.add_to_collection('S_average_cost', self.S_cost)


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
    VTlabel = data[batch_start + 1: batch_start + TIME_STEPS*BATCH_SIZE + 1, :-1].reshape((-1,TIME_STEPS,OUTPUT_SIZE))
    return [VTxin, VTlabel]

# 训练 LSTMRNN
if __name__ == '__main__':
    BATCH_START = 0  # 建立 batch data 时候的 index
    TIME_STEPS = 1  # backpropagation through time 的time_steps 误差反传的步长
    BATCH_SIZE = 500 #每次训练的步长的数量
    INPUT_SIZE = 6 # 数据输入size
    OUTPUT_SIZE = 5  # 数据输出 size
    D_INPUT_SIZE = 3  # 数据输入size
    D_OUTPUT_SIZE = 2  # 数据输出 size
    S_INPUT_SIZE = 6 # 数据输入size
    S_OUTPUT_SIZE = 3  # 数据输出 size
    CELL_SIZE = 64 # RNN的 hidden unit size
    LR = 0.005  # learning rate
    INITSTATE = False  # 判断是否读到数据的末尾，通过置零BATCHB_START开启第二轮训练
    MODEL_NAME = 'MIXED_MODEL'
    MODEL_ITERATION = '10000'
    # 加载数据
    buildData = dataRebuild('Data.txt', 0.7, 0.15)
    trainData = np.loadtxt('trainData.txt')
    testData = np.loadtxt('testData.txt')
    valData = np.loadtxt('valData.txt')
    # 搭建 LSTMRNN 模型
    model = LSTMRNN_D(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, D_INPUT_SIZE, D_OUTPUT_SIZE, S_INPUT_SIZE,   S_OUTPUT_SIZE,
                      CELL_SIZE, BATCH_SIZE, LR, norm = False)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=0.5)
    if os.path.exists('model') and os.path.exists('model/%s-%s.meta'%(MODEL_NAME, MODEL_ITERATION)) :
        print('restore last model')
        saver.restore(sess, 'model/%s-%s'%(MODEL_NAME, MODEL_ITERATION))
    else:
        print('create a new model')
        sess.run(tf.global_variables_initializer())


    # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
    # $ tensorboard --logdir='logs'

    # matplotlib可视化
    plt.ion()  # 设置连续 plot
    plt.show()
    D_fig = plt.figure(1, figsize=(10,12), dpi=80)
    S_fig = plt.figure(2, figsize=(10,12), dpi=80)
    iteration = 0
    D_cost_list = []
    S_cost_list = []
    D_trainLoss_list = []
    S_trainLoss_list = []
    D_testLoss_list = []
    S_testLoss_list = []
    D_valLoss_list = []
    S_valLoss_list = []
    D_label_list =[]
    S_label_list = []
    D_pred_list =[]
    S_pred_list =[]
    D_vallabel_list =[]
    S_vallabel_list = []
    D_valpred_list = []
    S_valpred_list =[]
    # 训练
    while(True):
        iteration += 1
        seq, res, xs = get_batch(trainData)  # 提取 batch data
        if iteration == 1 or INITSTATE:
            INITSTATE = False
            # 初始化 data
            feed_dict = {
                model.xs: seq,
                model.ys: res,
            }
        else:
            feed_dict = {
                model.xs: seq,
                model.ys: res,
                model.D_cell_init_state: D_state,  # 保持 state 的连续性
                model.S_cell_init_state: S_state
            }

        # 训练
        _D, _S, D_cost, S_cost, D_state, S_state, D_pred, S_pred = sess.run(
            [model.D_train_op, model.S_train_op, model.D_cost, model.S_cost,model.D_cell_final_state,
             model.S_cell_final_state, model.pred_D, model.pred_S],
            feed_dict=feed_dict)

        # 每50次迭代记录一次log 并print cost
        if iteration%50 == 0 :
            D_cost_list.append(D_cost)
            S_cost_list.append(S_cost)
            D_label_list.append(res.reshape(-1, OUTPUT_SIZE)[-1, 3:5])
            S_label_list.append(res.reshape(-1, OUTPUT_SIZE)[-1, :3])
            D_pred_list.append(D_pred.reshape(-1, OUTPUT_SIZE)[-1, :])
            S_pred_list.append(S_pred.reshape(-1, OUTPUT_SIZE)[-1, :])
            if len(D_trainLoss_list) == 300:
                D_valLoss_list = []
                S_valLoss_list = []
                DtrainLoss_list = []
                StrainLoss_list = []
                # testLoss_list = []
            VD = get_VTData(trainData)
            valxs = VD[0]
            valys = VD[1]
            D_valLoss, D_valpre, S_valLoss, S_valpre = sess.run([model.D_cost, model.pred_D, model.S_cost, model.pred_S], feed_dict={model.xs: valxs,model.ys: valys})
            D_vallabel_list.append(valys.reshape(-1, OUTPUT_SIZE)[-1, 3:5])
            S_vallabel_list.append(valys.reshape(-1, OUTPUT_SIZE)[-1, :3])
            D_valpred_list.append(D_valpre.reshape(-1, OUTPUT_SIZE)[-1, :])
            S_valpred_list.append(S_valpre.reshape(-1, OUTPUT_SIZE)[-1, :])
            # cost保留四位小数输出
            print('iteration: ', iteration, 'D_train_cost: ', round(D_cost, 4), 'D_val_cost:', round(D_valLoss, 4),
                  'D_test_cost:', 'None\n',
                  '\t\t\t\t','S_train_cost: ', round(S_cost, 4), 'S_val_cost:', round(S_valLoss, 4),
                  'S_test_cost:', 'None'
                  )
            # # # 测试验证模型
            # testLoss_list.append(testLoss)
            D_valLoss_list.append(D_valLoss)
            S_valLoss_list.append(S_valLoss)
            D_trainLoss_list.append(np.mean(D_cost_list))
            S_trainLoss_list.append(np.mean(S_cost_list))
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, iteration)


        # 每1000次迭代保存一次模型 并打印cost曲线
        if iteration % 2500 == 0:
            # # # 保存模型
            saver.save(sess, 'model/%s'%MODEL_NAME, global_step=iteration)

            # # # 绘制数据动态图
            S_label = np.array(S_label_list)
            S_output = np.array(S_pred_list)
            S_vallabel = np.array(S_vallabel_list)
            S_valOutput = np.array(S_valpred_list)
            plt.clf()
            S_chart_1 = S_fig.add_subplot(5, 1, 1, xlim=(0, S_label.shape[0]))
            S_chart_1.set_title('S_L')
            S_chart_2 = S_fig.add_subplot(5, 1, 2, xlim=(0, S_label.shape[0]))
            S_chart_2.set_title('S_M')
            S_chart_3 = S_fig.add_subplot(5, 1, 3, xlim=(0, S_label.shape[0]))
            S_chart_3.set_title('S_R')
            S_chart_4 = S_fig.add_subplot(5, 1, 4, xlim=(0, len(S_cost_list)))
            S_chart_4.set_title('S_trainLoss')
            S_chart_5 = S_fig.add_subplot(5, 1, 5, xlim=(0, len(S_trainLoss_list)))
            S_chart_5.set_title('S_train_test_val_loss')


            D_label = np.array(D_label_list)
            D_output = np.array(D_pred_list)
            D_vallabel = np.array(D_vallabel_list)
            D_valOutput = np.array(D_valpred_list)
            D_chart_1 = D_fig.add_subplot(4, 1, 1, xlim=(0, D_label.shape[0]))
            D_chart_1.set_title('Dis_1')
            D_chart_2 = D_fig.add_subplot(4, 1, 2, xlim=(0, D_label.shape[0]))
            D_chart_2.set_title('Dis_2')
            D_chart_3 = D_fig.add_subplot(4, 1, 3, xlim=(0, len(D_cost_list)))
            D_chart_3.set_title('D_trainLoss')
            D_chart_4 = D_fig.add_subplot(4, 1, 4, xlim=(0, len(D_trainLoss_list)))
            D_chart_4.set_title('D_train_test_val_loss')

            # S_L
            S_chart_1.scatter(np.arange(0, S_label.shape[0]), S_label[:, 0], c='blue', s=10, marker='o')
            S_chart_1.scatter(np.arange(0, S_output.shape[0]), S_output[:, 0], c='red', s=10, marker='x')
            # S_M
            S_chart_2.scatter(np.arange(0, S_label.shape[0]), S_label[:, 1], c='blue', s=10, marker='o')
            S_chart_2.scatter(np.arange(0, S_output.shape[0]), S_output[:, 1], c='red', s=10, marker='x')
            # S_R
            S_chart_3.scatter(np.arange(0, S_label.shape[0]), S_label[:, 2], c='blue', s=10, marker='o')
            S_chart_3.scatter(np.arange(0, S_output.shape[0]), S_output[:, 2], c='red', s=10, marker='x')
            # Cost
            S_chart_4.plot(np.arange(0, len(S_cost_list)), S_cost_list, 'r-', lw=2)
            # VT_cost
            # chart_7.plot (np.arange(0, len(testLoss_list)), testLoss_list, 'g-', lw = 2)
            S_chart_5.plot(np.arange(0, len(S_valLoss_list)), S_valLoss_list, 'r--', lw=2)
            S_chart_5.plot(np.arange(0, len(S_trainLoss_list)), S_trainLoss_list, 'b--', lw=2)

            # Dis_1
            D_chart_1.plot(np.arange(0, D_label.shape[0]), D_label[:, 0], 'b-', lw=4)
            D_chart_1.plot(np.arange(0, D_output.shape[0]), D_output[:, 0], 'r--', lw=2)
            # Dis_2
            D_chart_2.plot(np.arange(0, D_label.shape[0]), D_label[:, 1], 'b-', lw=4)
            D_chart_2.plot(np.arange(0, D_output.shape[0]), D_output[:, 1], 'r--', lw=2)
            # Cost
            D_chart_3.plot(np.arange(0, len(D_cost_list)), D_cost_list, 'r-', lw=2)
            # VT_cost
            # chart_7.plot (np.arange(0, len(testLoss_list)), testLoss_list, 'g-', lw = 2)
            D_chart_4.plot(np.arange(0, len(D_valLoss_list)), D_valLoss_list, 'r--', lw=2)
            D_chart_4.plot(np.arange(0, len(D_trainLoss_list)), D_trainLoss_list, 'b--', lw=2)
            plt.pause(0.1)
            if not os.path.exists('plot'):
                os.makedirs('plot')
            plt.savefig('plot/iteration_%s.png' % iteration)
            D_cost_list = []
            S_cost_list = []
            D_label_list = []
            S_label_list = []
            D_pred_list = []
            S_pred_list = []
            D_vallabel_list = []
            S_vallabel_list = []
            D_valpred_list = []
            S_valpred_list = []

