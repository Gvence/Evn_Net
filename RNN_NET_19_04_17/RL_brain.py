"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import signal
import time

#np.random.seed(1)
#tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions = 3,
            n_features = 5,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=1000,
            memory_size=5000,
            stack_size = 10,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.stack_size = stack_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.saving_stack = False
        self.read_OK = True
        self.INITSTATE = True
        self.FIND_MEMORY_STACK =os.path.exists('memory_stack.txt')
        self.cost_his = []
        self.MODEL_NAME = 'DQN_MODEL'
        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.Evn_memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.M_Evn_memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.memory_stack = {}

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.graph = tf.get_default_graph()
        self.sess = tf.Session(graph=self.graph)

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)
        with self.sess.as_default():
            with self.graph.as_default():
                self.sess.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver(max_to_keep=4)

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_,Evn = False, M_Evn = False):
        transition = np.hstack((s, [a, r], s_))
        #保存实际环境数据
        if Evn :
            # 判断self对象是否存在某个属性或方法
            if not hasattr(self, 'Evn_memory_counter'):
                self.Evn_memory_counter = 0
            # replace the old memory with new memory
            self.Evn_memory[self.Evn_memory_counter, :] = transition
            self.Evn_memory_counter += 1
            #存满一个size保存一次
            if self.Evn_memory_counter % self.memory_size == 0 :
                self.Evn_memory_counter = 0
                self.memory_management()
        if M_Evn :
            if not hasattr(self, 'M_Evn_memory_counter'):
                self.M_Evn_memory_counter = 0
            self.M_Evn_memory[self.M_Evn_memory_counter, :]=transition
            self.M_Evn_memory_counter += 1
            if self.M_Evn_memory_counter%self.memory_size == 0:
                self.M_Evn_memory_counter = 0

    def memory_management(self):
        if not hasattr(self, 'stack_len'):
            self.stack_len = 0
        self.memory_stack[str(self.stack_len)] = self.Evn_memory.tolist()
        print('push shape %s M in stack[\'%s\']'%(self.Evn_memory.shape, self.stack_len))
        self.stack_len += 1
        if self.stack_len % self.stack_size == 0:
            self.stack_len = 0
            self.saving_stack = True
            print('saving memory stack...')
            file = open('memory_stack.txt','w')
            file.write(str(self.memory_stack))
            file.close()
            self.saving_stack = False
            self.FIND_MEMORY_STACK = True

    def read_memorystack(self):
        self.read_OK = False
        while not self.read_OK:
                if os.path.exists('memory_stack.txt'):
                    print('reading memory srtack...')
                    file = open ('memory_stack.txt','r')
                    result = file.read()
                    dic = eval(result)
                    file.close()
                    self.read_OK = True
        return dic

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = np.array(observation).reshape(-1,self.n_features)

        if np.random.uniform() < self.epsilon:
            # with self.sess.as_default():
            #     with self.graph.as_default():
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')
            if self.learn_step_counter != 0:
                #self.saver.save(self.sess,os.path.join(os.getcwd(), 'model/dqn_model/%s'%self.MODEL_NAME), global_step=int(self.learn_step_counter/self.replace_target_iter))
                self.saver.save(self.sess,'model/dqn_model/%s'%self.MODEL_NAME, global_step=int(self.learn_step_counter/self.replace_target_iter))

        # sample batch memory from all memory
        sample_index = np.random.choice(len(self.M_Evn_memory), size=self.batch_size)
        batch_memory = self.M_Evn_memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        batch_index 的目的是生成一个batch中所有的状态标签，[0,1,2......n]从第0状态到第n状态
        
        eval_act_index 是为了将某个s下的动作a记录下来，在这里，a有三种[0,1,2],随后在计算出q_target后
        用这些index找出某个状态下选取的那个动作，并把target值付给他
        """

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



