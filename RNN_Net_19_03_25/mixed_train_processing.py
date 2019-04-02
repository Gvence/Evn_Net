#coding=utf-8
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

    # 重新选取片段或者其实数据达到临界值时重新选取起始值
    while  INITSTATE:
        BATCH_START = random.randint(0, len(trainData) - 1 - 2 * TIME_STEPS * init_size - BATCH_SIZE)
        INITSTATE = False
    BATCH_START += 1
    xs = trainData[BATCH_START:BATCH_START + TIME_STEPS*init_size,  :].reshape((-1,TIME_STEPS ,INPUT_SIZE))
    ys = trainData[BATCH_START + 1:BATCH_START + TIME_STEPS*init_size + 1, :].reshape((-1,TIME_STEPS,INPUT_SIZE))
    return [xs, ys]

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
                #self.D_l_in_y = self.normalize(self.D_l_in_y, 0, self.cell_size)
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
            #if self.norm :
                #self.D_cell_outputs = self.normalize(self.D_cell_outputs, 0, self.cell_size)
            tf.summary.histogram('rnn_out', self.D_cell_outputs)

        with tf.name_scope('S_RNN_cell'):
            S_lstm_cell = tf.nn.rnn_cell.LSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True, name='S_RNN_cell')
            with tf.name_scope('S_initial_state'):
                self.S_cell_init_state = S_lstm_cell.zero_state(self.batch_size, dtype=tf.float32)

            self.S_cell_outputs, self.S_cell_final_state = tf.nn.dynamic_rnn(
                S_lstm_cell, self.S_l_in_y, initial_state=self.S_cell_init_state, time_major=False, )
            if self.norm:
                self.S_cell_outputs = self.normalize(self.S_cell_outputs, 0, self.cell_size)
                #self.D_cell_outputs = self.normalize(self.D_cell_outputs, 0, self.cell_size)
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
    global batch_start, val_init
    while val_init:
        batch_start = random.randint(0, len(data) - 1 - 2 * TIME_STEPS*init_size - BATCH_SIZE)
        val_init = False
    batch_start +=1
    VTxin = data[batch_start:batch_start + TIME_STEPS*init_size , :].reshape((-1,TIME_STEPS ,INPUT_SIZE))
    VTlabel = data[batch_start + 1: batch_start + TIME_STEPS*init_size + 1, :].reshape((-1,TIME_STEPS,INPUT_SIZE))
    return [VTxin, VTlabel]

# 训练 LSTMRNN
if __name__ == '__main__':
    BATCH_START = []  # 建立 batch data 时候的 index
    TIME_STEPS =1 # backpropagation through time 的time_steps 误差反传的步长
    init_size = 1
    BATCH_SIZE = 100 #每次训练的步长的数量
    INPUT_SIZE = 6 # 数据输入size
    OUTPUT_SIZE = 5  # 数据输出 size
    D_INPUT_SIZE = 3  # 数据输入size
    D_OUTPUT_SIZE = 2  # 数据输出 size
    S_INPUT_SIZE = 6 # 数据输入size
    S_OUTPUT_SIZE = 3  # 数据输出 size
    CELL_SIZE = 32 # RNN的 hidden unit size
    LR = 0.006  # learning rate
    INITSTATE = False  # 判断是否读到数据的末尾，通过置零BATCHB_START开启第二轮训练
    MODEL_NAME = 'MIXED_MODEL'
    MODEL_ITERATION = '0'
    # 加载数据
    buildData = dataRebuild('Data.txt', 0.9, 0.05)
    trainData = np.loadtxt('trainData.txt')
    DATA_LONG = np.shape(trainData)[0]
    testData = np.loadtxt('testData.txt')
    valData = np.loadtxt('valData.txt')
    # 搭建 LSTMRNN 模型
    model = LSTMRNN_D(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, D_INPUT_SIZE, D_OUTPUT_SIZE, S_INPUT_SIZE,   S_OUTPUT_SIZE,
                      CELL_SIZE, init_size, LR, norm = True)
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
    fig = plt.figure(1, figsize=(24,12), dpi=80)

    iteration = 0
    times_tip = 0
    D_trainLoss_list = []
    S_trainLoss_list = []
    D_valLoss_list = []
    S_valLoss_list = []

    # 训练
    while(True):
        #每个训练周期更新一次输入
        if times_tip % BATCH_SIZE == 0:
            iteration += 1
            INITSTATE = True
            #训练记录表清空
            D_cost_list = []
            S_cost_list = []
            D_label_list = []
            S_label_list = []
            D_pred_list = []
            S_pred_list = []
            right_times = 0
            # 初始化提取 batch data， 给出初始状态链
            S, S_= get_batch(trainData)
            if times_tip == 0:
                feed_dict = {
                model.xs: S,
                model.ys: S_[:,:,:-1]
                }
            elif times_tip >0 :
                feed_dict = {
                    model.xs: S,
                    model.ys: S_[:, :, :-1],
                    model.D_cell_init_state: D_state,  # 保持 state 的连续性
                    model.S_cell_init_state: S_state
                }
            else:
                print('OMG iteration less than 0 !!! something wrong~~')
        else:
            #上回合预测出的状态下，做的动作，由上回合标签状态连的最后一个决定
            action = S_[-1,-1,-1]
            #更新下个状态的label
            S_ = get_batch(trainData)[1]

            #将上回合的状态预测作为一个状态嵌入到状态链的末尾，形成一个新的状态链（状态连迭代）
            S = np.concatenate((S.reshape(-1, INPUT_SIZE)[1:, :],np.concatenate((car_S, [action])).reshape(-1, INPUT_SIZE))).reshape(-1, TIME_STEPS, INPUT_SIZE)
            feed_dict = {
                model.xs: S,
                model.ys: S_[:,:,:-1],
                model.D_cell_init_state: D_state,  # 保持 state 的连续性
                model.S_cell_init_state: S_state
            }

        # 训练
        _D, _S, \
        D_cost, S_cost, \
        D_state, S_state, \
        D_pred, S_pred = sess.run(
            [model.D_train_op, model.S_train_op,
             model.D_cost, model.S_cost,
             model.D_cell_final_state,mod.el.S_cell_final_state,
             model.pred_D, model.pred_S],
            feed_dict=feed_dict)
        # 将训练结果拼接，作为下个状态的预测
        if S_cost <0.1 and  D_cost < 1000 :
            car_S = np.concatenate((np.round(S_pred.reshape(-1, S_OUTPUT_SIZE)[-1, :],0), D_pred.reshape(-1, D_OUTPUT_SIZE)[-1, :]))
            right_times += 1
        else:
            #right_times = 0
            car_S = S_[-1,-1,:-1]
        # 记录误差
        D_cost_list.append(D_cost)
        S_cost_list.append(S_cost)
        D_label_list.append(S_.reshape(-1, OUTPUT_SIZE + 1)[-1, 3:-1])
        S_label_list.append(S_.reshape(-1, OUTPUT_SIZE + 1)[-1, :3])
        D_pred_list.append(D_pred.reshape(-1, D_OUTPUT_SIZE)[-1, :])
        S_pred_list.append(S_pred.reshape(-1, S_OUTPUT_SIZE)[-1, :])
        times_tip += 1

        # 每20次epoch记录一次log 并print
        if times_tip%(BATCH_SIZE*20 ) == 0 :
            #列表初始化
            val_D_cost_list = []
            val_S_cost_list = []
            val_D_label_list = []
            val_S_label_list = []
            val_D_pred_list = []
            val_S_pred_list = []

            val_init = True
            times = 0

            while times <= BATCH_SIZE:
                if val_init:
                    val_right_times = 0
                    val_S, val_S_ = get_VTData(valData)
                    val_feed_dict = {model.xs: val_S, model.ys:val_S_[:,:,:-1]}
                else:
                    action = val_S_[-1, -1, -1]
                    val_S_ = get_VTData(valData)[1]
                    val_S = np.concatenate((val_S.reshape(-1, INPUT_SIZE)[1:, :],np.concatenate((val_car_S, [action])).reshape(-1, INPUT_SIZE))).reshape(-1, TIME_STEPS, INPUT_SIZE)
                    feed_dict = {
                        model.xs: val_S,
                        model.ys: val_S_[:,:,:-1]
                        # model.D_cell_init_state:val_D_state,
                        # model.S_cell_init_state:val_S_state
                    }
                val_D_cost, val_S_cost, \
                val_D_pred, val_S_pred = sess.run([model.D_cost, model.S_cost,
                                                # model.D_cell_final_state,model.S_cell_final_state,
                                                model.pred_D, model.pred_S], feed_dict=val_feed_dict)
                if val_D_cost < 1000 and val_S_cost < 0.1 :
                    val_car_S = np.concatenate((np.round(val_S_pred.reshape(-1, S_OUTPUT_SIZE)[-1, :],0), val_D_pred.reshape(-1, D_OUTPUT_SIZE)[-1, :]))
                    val_right_times += 1
                else:
                    #val_right_times = 0
                    val_car_S = val_S_[-1,-1,:-1]
                val_D_cost_list.append(val_D_cost)
                val_S_cost_list.append(val_S_cost)
                val_D_label_list.append(val_S_.reshape(-1, OUTPUT_SIZE + 1)[-1, 3:-1])
                val_S_label_list.append(val_S_.reshape(-1, OUTPUT_SIZE + 1)[-1, : 3])
                val_D_pred_list.append(val_D_pred.reshape(-1, D_OUTPUT_SIZE)[-1, :])
                val_S_pred_list.append(val_S_pred.reshape(-1, S_OUTPUT_SIZE)[-1, :])
                times +=1

            #记录最近一次迭代的平均误差
            D_mean_cost = np.mean(np.array(D_cost_list))
            S_mean_cost = np.mean(np.array(S_cost_list))
            val_D_mean_cost = np.mean(np.array(val_D_cost_list))
            val_S_mean_cost = np.mean(np.array(val_S_cost_list))
            # cost保留四位小数输出
            print('EPO:',int((iteration*BATCH_SIZE*init_size*TIME_STEPS)/(int(DATA_LONG/BATCH_SIZE)*BATCH_SIZE)),'ITER: ', iteration, 'D_train_cost: ', D_mean_cost, 'S_train_cost:', round(S_mean_cost, 4),
                  'Accuraty:%.2f%%\n'%(100*right_times/BATCH_SIZE),
                  '\t\t\t\t','D_val_cost: ', round(val_D_mean_cost, 4), 'S_val_cost:', round(val_S_mean_cost, 4),
                  'Val_Accuraty:%.2f%%\n'%(100*val_right_times/BATCH_SIZE)
                  )

            # 用于交叉验证的误差列表
            D_valLoss_list.append(val_D_mean_cost)
            S_valLoss_list.append(val_S_mean_cost)
            D_trainLoss_list.append(D_mean_cost)
            S_trainLoss_list.append(S_mean_cost)

            #保存一次TFlog
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, iteration)


            # # # 保存模型
            saver.save(sess, 'model/%s'%MODEL_NAME, global_step=iteration)


            # # # 绘制数据动态图
            S_label = np.array(S_label_list)
            S_output = np.array(S_pred_list)
            S_vallabel = np.array(val_S_label_list)
            S_valoutput = np.array(val_S_pred_list)
            plt.clf()
            S_chart_1 = fig.add_subplot(8, 2, 1, xlim=(0, S_label.shape[0]))
            S_chart_1.set_title('S_L')
            S_chart_2 = fig.add_subplot(8, 2, 3, xlim=(0, S_label.shape[0]))
            S_chart_2.set_title('S_M')
            S_chart_3 = fig.add_subplot(8, 2, 5, xlim=(0, S_label.shape[0]))
            S_chart_3.set_title('S_R')
            S_chart_4 = fig.add_subplot(8, 2, 7, xlim=(0, S_vallabel.shape[0]))
            S_chart_4.set_title('VAL_S_L')
            S_chart_5 = fig.add_subplot(8, 2, 9, xlim=(0, S_vallabel.shape[0]))
            S_chart_5.set_title('VAL_S_M')
            S_chart_6 = fig.add_subplot(8, 2, 11, xlim=(0, S_vallabel.shape[0]))
            S_chart_6.set_title('VAL_S_R')
            S_chart_7 = fig.add_subplot(8, 2, 13, xlim=(0, len(S_trainLoss_list)))
            S_chart_7.set_title('S_trainLoss')
            S_chart_8 = fig.add_subplot(8, 2, 15, xlim=(0, len(S_valLoss_list)))
            S_chart_8.set_title('S_train_test_val_loss')


            D_label = np.array(D_label_list)
            D_output = np.array(D_pred_list)
            D_vallabel = np.array(val_D_label_list)
            D_valoutput = np.array(val_D_pred_list)

            D_chart_1 = fig.add_subplot(8, 2, 2, xlim=(0, D_label.shape[0]))
            D_chart_1.set_title('Dis_1')
            D_chart_2 = fig.add_subplot(8, 2, 4, xlim=(0, D_label.shape[0]))
            D_chart_2.set_title('Dis_2')
            D_chart_3 = fig.add_subplot(8, 2, 6, xlim=(0, D_label.shape[0]))
            D_chart_3.set_title('VALDis_1')
            D_chart_4 = fig.add_subplot(8, 2, 8, xlim=(0, D_label.shape[0]))
            D_chart_4.set_title('VALDis_2')
            D_chart_5 = fig.add_subplot(8, 2, 10, xlim=(0, len(D_trainLoss_list)))
            D_chart_5.set_title('D_trainLoss')
            D_chart_6 = fig.add_subplot(8, 2, 12, xlim=(0, len(D_valLoss_list)))
            D_chart_6.set_title('D_train_test_val_loss')

            # S_L
            S_chart_1.scatter(np.arange(0, S_label.shape[0]), S_label[:, 0], c='blue', s=10, marker='o')
            S_chart_1.scatter(np.arange(0, S_output.shape[0]), S_output[:, 0], c='red', s=10, marker='x')
            # S_M
            S_chart_2.scatter(np.arange(0, S_label.shape[0]), S_label[:, 1], c='blue', s=10, marker='o')
            S_chart_2.scatter(np.arange(0, S_output.shape[0]), S_output[:, 1], c='red', s=10, marker='x')
            # S_R
            S_chart_3.scatter(np.arange(0, S_label.shape[0]), S_label[:, 2], c='blue', s=10, marker='o')
            S_chart_3.scatter(np.arange(0, S_output.shape[0]), S_output[:, 2], c='red', s=10, marker='x')
            # VALS_L
            S_chart_4.scatter(np.arange(0, S_vallabel.shape[0]), S_vallabel[:, 0], c='blue', s=10, marker='o')
            S_chart_4.scatter(np.arange(0, S_valoutput.shape[0]), S_valoutput[:, 0], c='red', s=10, marker='x')
            # VALS_M
            S_chart_5.scatter(np.arange(0, S_vallabel.shape[0]), S_vallabel[:, 1], c='blue', s=10, marker='o')
            S_chart_5.scatter(np.arange(0, S_valoutput.shape[0]), S_valoutput[:, 1], c='red', s=10, marker='x')
            # VALS_R
            S_chart_6.scatter(np.arange(0, S_vallabel.shape[0]), S_vallabel[:, 2], c='blue', s=10, marker='o')
            S_chart_6.scatter(np.arange(0, S_valoutput.shape[0]), S_valoutput[:, 2], c='red', s=10, marker='x')
            # Cost
            S_chart_7.plot(np.arange(0, len(S_trainLoss_list)), S_trainLoss_list, 'r-', lw=2)
            # VT_cost
            S_chart_8.plot(np.arange(0, len(S_valLoss_list)), S_valLoss_list, 'r--', lw=2)
            S_chart_8.plot(np.arange(0, len(S_trainLoss_list)), S_trainLoss_list, 'b--', lw=2)

            # Dis_1
            D_chart_1.scatter(np.arange(0, D_label.shape[0]), D_label[:, 0], c='blue', s=10, marker='o')
            D_chart_1.scatter(np.arange(0, D_output.shape[0]), D_output[:, 0], c='red', s=10, marker='x')
            # Dis_2
            D_chart_2.scatter(np.arange(0, D_label.shape[0]), D_label[:, 1], c='blue', s=10, marker='o')
            D_chart_2.scatter(np.arange(0, D_output.shape[0]), D_output[:, 1], c='red', s=10, marker='x')
            # VALDis_1
            D_chart_3.scatter(np.arange(0, D_vallabel.shape[0]), D_vallabel[:, 0],  c='blue', s=10, marker='o')
            D_chart_3.scatter(np.arange(0, D_valoutput.shape[0]), D_valoutput[:, 0], c='red', s=10, marker='x')
            # VALDis_2
            D_chart_4.scatter(np.arange(0, D_vallabel.shape[0]), D_vallabel[:, 1],  c='blue', s=10, marker='o')
            D_chart_4.scatter(np.arange(0, D_valoutput.shape[0]), D_valoutput[:, 1], c='red', s=10, marker='x')
            # Cost
            D_chart_5.plot(np.arange(0, len(D_trainLoss_list)), D_trainLoss_list, 'r-', lw=2)
            # VT_cost
            D_chart_6.plot(np.arange(0, len(D_valLoss_list)), D_valLoss_list, 'r--', lw=2)
            D_chart_6.plot(np.arange(0, len(D_trainLoss_list)), D_trainLoss_list, 'b--', lw=2)
            plt.pause(0.1)
            if not os.path.exists('plot'):
                os.makedirs('plot')
            plt.savefig('plot/iteration_%s.png' % (iteration))
            if len(D_trainLoss_list) == 100:
                D_trainLoss_list = []
                S_trainLoss_list = []
                D_valLoss_list = []
                S_valLoss_list = []
