#coding=utf-8
import tensorflow as tf
import os
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from Data_Processing import dataRebuild
from RL_brain import DeepQNetwork
import random

# 定义 LSTMRNN 的主体结构
class LSTMRNN(object):
    def __init__(self, n_steps = 1, input_size = 6, output_size = 5,
                 D_input_size = 3, D_output_size = 2, S_input_size = 6,
                 S_output_size = 3, cell_size = 64, batch_size = 500, learning_rate = 0.006,
                 step_length =500 ,norm = False, name = 'MIXED_EVN_MODEL', iteration = 0,ploter = True):
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
        self.step_length = step_length
        self.MODEL_NAME = name
        self.iteration = iteration
        self.ploter = ploter
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
        self.init_model()

    #初始化模型，检查是否已经存在模型，如果存在可以根据调整iteration使模型接着上个iteration训练，如果iteration设置0，则重新训练
    def init_model(self):
        self.INITSTATE = True
        self.TRAINING = False
        self.times_tip = 0
        self.graph =tf.get_default_graph()
        self.sess = tf.Session(graph=self.graph)
        with self.sess.as_default():
            with self.graph.as_default():
                print('create a new model')
                self.sess.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver(max_to_keep=4)
                if os.path.exists('model/evn_model') and os.path.exists('model/evn_model/%s-%s.meta' % (self.MODEL_NAME, str(self.iteration))):
                    print('restore last model')
                    self.saver.restore(self.sess, 'model/evn_model/%s-%s' % (self.MODEL_NAME, str(self.iteration)))
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("logs", self.sess.graph)
        if self.ploter :
            plt.ion()
            plt.show()
            self.fig = plt.figure(1, figsize=(24,12), dpi=80)


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
            # with tf.name_scope('D_initial_state'):
            #     self.D_cell_init_state = D_lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            self.D_cell_outputs, self.D_cell_final_state = tf.nn.dynamic_rnn(
                D_lstm_cell, self.D_l_in_y, dtype = tf.float32, time_major=False, )
            #if self.norm :
                #self.D_cell_outputs = self.normalize(self.D_cell_outputs, 0, self.cell_size)
            tf.summary.histogram('rnn_out', self.D_cell_outputs)

        with tf.name_scope('S_RNN_cell'):
            S_lstm_cell = tf.nn.rnn_cell.LSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True, name='S_RNN_cell')
            # with tf.name_scope('S_initial_state'):
            #     self.S_cell_init_state = S_lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            self.S_cell_outputs, self.S_cell_final_state = tf.nn.dynamic_rnn(
                S_lstm_cell, self.S_l_in_y, dtype = tf.float32, time_major=False, )
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

    #从memory中抽取一个batch用于训练
    def get_batch(self, trainData):

        #初始华时，或者当起始点后推至临界时重新选取起始值
        if self.INITSTATE:
            self.Data = np.array(trainData)
            self.batch_start= np.array([])
            self.batch_start = np.random.choice((len(self.Data) - 1 - 2*self.n_steps -self.step_length), self.batch_size)

            #抽取结束后关闭初始化模式
            self.INITSTATE = False
        else:
            #每次抽取训练数据起始点自动后移一个time_point
            self.batch_start += 1

            #清空训练数据list
            self.input = np.array([])
            self.target = np.array([])

            #读取每个起点的数据
            for j in self.batch_start:
                j = int(j)
                self.input = np.append(self.input, self.Data[j : j + self.n_steps, :6])
                self.target = np.append(self.target, self.Data[j: j + self.n_steps, -5:])
            self.input = self.input.reshape((-1, self.n_steps, self.input_size))
            self.target = self.target.reshape((-1, self.n_steps, self.output_size))

    #训练网络
    def train(self):

        #开是训练
        self.TRAINING = True
        #print('train')

        #每完成step_length次训练，初始一次记录list
        if self.times_tip%self.step_length == 0:
            self.iteration += 1
            self.D_cost_list = []
            self.S_cost_list = []
            self.D_label_list = []
            self.S_label_list = []
            self.D_pred_list = []
            self.S_pred_list = []

        #获取一个time point 的input 和 target
        self.get_batch(self.Data)

        feed_dict = {
                self.xs:self.input,
                self.ys:self.target
            }

        #训练
        _D, _S, \
        D_cost, S_cost, \
        D_pred, S_pred = self.sess.run(
            [self.D_train_op, self.S_train_op,
             self.D_cost, self.S_cost,
             self.pred_D, self.pred_S],
            feed_dict=feed_dict)

        # 记录误差
        self.D_cost_list.append(D_cost)
        self.S_cost_list.append(S_cost)


        #一次训练结束
        self.times_tip +=1


        #如果完成step_length次训练，开始初始数据
        if self.times_tip%self.step_length == 0:

            # 保存log
            self.result = self.sess.run(self.merged, feed_dict)
            self.writer.add_summary(self.result, self.iteration)

            # 保存 model
            if self.times_tip%self.step_length *100 == 0:

            # 保存每个iteration最后一次训练的数据
                index = np.random.choice((self.batch_size), int(self.batch_size / 5))
                self.D_label_list.append(self.target[list(index), -1, :].reshape(-1, self.output_size)[:, 3:-1])
                self.S_label_list.append(self.target[list(index), -1, :].reshape(-1, self.output_size)[:, :3])
                self.D_pred_list.append(D_pred.reshape(-1, self.n_steps, self.D_output_size)
                                        [list(index), -1, :].reshape(-1, self.D_output_size))
                self.S_pred_list.append(S_pred.reshape(-1, self.n_steps, self.S_output_size)
                                        [list(index), -1, :].reshape(-1, self.S_output_size))
                print('saving Evn model %s' % self.iteration)
                self.saver.save(self.sess, 'model/evn_model/%s'%self.MODEL_NAME, global_step=self.iteration)
                self.plotting()
            self.INITSTATE = True

        self.TRAINING = False

    #绘制训练参数
    def plotting(self):
        #清空窗口
        plt.clf()

        S_label = np.array(self.S_label_list).reshape(-1,self.S_output_size)
        S_pred = np.array(self.S_pred_list).reshape(-1,self.S_output_size)
        S_chart_1 = self.fig.add_subplot(4, 2, 1, xlim=(0, S_label.shape[0]))
        S_chart_1.set_title('S_L')
        S_chart_2 = self.fig.add_subplot(4, 2, 3, xlim=(0, S_label.shape[0]))
        S_chart_2.set_title('S_M')
        S_chart_3 = self.fig.add_subplot(4, 2, 5, xlim=(0, S_label.shape[0]))
        S_chart_3.set_title('S_R')
        S_chart_4 = self.fig.add_subplot(4, 2, 7, xlim=(0, len(self.S_cost_list)))
        S_chart_4.set_title('S_COST')

        D_label = np.array(self.D_label_list).reshape(-1,self.D_output_size)
        D_pred = np.array(self.D_pred_list).reshape(-1,self.D_output_size)
        D_chart_1 = self.fig.add_subplot(4, 2, 2, xlim=(0, D_label.shape[0]))
        D_chart_1.set_title('DIS1')
        D_chart_2 = self.fig.add_subplot(4, 2, 4, xlim=(0, D_label.shape[0]))
        D_chart_2.set_title('DIS2')
        D_chart_3 = self.fig.add_subplot(4, 2, 6, xlim=(0, len(self.D_cost_list)))
        D_chart_3.set_title('D_COST')

        #S_L
        S_chart_1.scatter(np.arange(0, S_label.shape[0]), S_label[:, 0], c='blue', s=10, marker='o')
        S_chart_1.scatter(np.arange(0, S_pred.shape[0]), S_pred[:, 0], c='red', s=10, marker='x')
        #S_M
        S_chart_2.scatter(np.arange(0, S_label.shape[0]), S_label[:, 1], c='blue', s=10, marker='o')
        S_chart_2.scatter(np.arange(0, S_pred.shape[0]), S_pred[:, 1], c='red', s=10, marker='x')
        #S_R
        S_chart_3.scatter(np.arange(0, S_label.shape[0]), S_label[:, 2], c='blue', s=10, marker='o')
        S_chart_3.scatter(np.arange(0, S_pred.shape[0]), S_pred[:, 2], c='red', s=10, marker='x')
        #S_COST
        S_chart_4.plot(np.arange(0, len(self.S_cost_list)), self.S_cost_list, 'r-', lw=2)

        #DIS1
        D_chart_1.plot(np.arange(0,D_label.shape[0]), D_label[:, 0], 'b-', lw=5)
        D_chart_1.plot(np.arange(0, D_pred.shape[0]), D_pred[:, 0], 'r-', lw=2)
        #DIS2
        D_chart_2.plot(np.arange(0, D_label.shape[0]), D_label[:, 1], 'b-', lw=5)
        D_chart_2.plot(np.arange(0, D_pred.shape[0]), D_pred[:, 1], 'r-', lw=2)
        #D_COST
        D_chart_3.plot(np.arange(0, len(self.D_cost_list)), self.D_cost_list, 'r-', lw=2)
        plt.pause(0.1)
        if not os.path.exists('plot'):
            os.makedirs('plot')
        plt.savefig('plot/iteration_%s.png' % self.iteration)



    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1., )
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)


