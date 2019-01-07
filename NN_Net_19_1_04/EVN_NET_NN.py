import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import random

#定义训练类
class trainOP():
    def __init__(self, filename, traiName = None):
        self.data = self.loadData(filename)
        self.traiName = traiName
        self.Iteration = 0
        self.ITE = []
    #随机函数
    def random(self, a, b):
        return random.randint(a, b)

    #导入数据
    def loadData(self, filename):
        return np.loadtxt(filename)
    #参数归一化
    def normalization(self, Data):
        m, n = np.shape(Data)
        Result = []
        for i in range(n):
            if i <=2 :
                max = 1
                min = 0
                avg = 0.5
            else :
                max = Data[:, i].max()
                min = Data[:, i].min()
                avg = np.mean(Data[:, i], axis=0)
            Result.append((Data[:, i] - avg)/(max - min))
        return np.array(Result).T
    #挑选数据
    def pickData(self):
        # a = self.random(0, (np.shape(self.data)[0] - 101))

        return self.data[:, :]

    def reshapData(self):
        Data = self.pickData()
        SL_max = 1
        SL_min = 0
        SL_avg = np.mean(Data[:, 0], axis = 0)

        SM_max = 1
        SM_min = 0
        SM_avg = np.mean(Data[:, 1], axis = 0)

        SR_max = 1
        SR_min = 0
        SR_avg = np.mean(Data[:, 2], axis = 0)

        Dis_max = Data[:, 3].max()
        Dis_min = Data[:, 3].min()
        Dis_avg = np.mean(Data[1:, 3], axis = 0)

        # Rew_max = Data[1:, 4].max()
        # Rew_min = Data[1:, 4].min()
        # Rew_avg = np.mean(Data[1:101, 4], axis = 0)

        Act_max = Data[:, 5].max()
        Act_min = Data[:, 5].min()
        Act_avg = np.mean(Data[1:, 5], axis = 0)

        Data = self.normalization(Data)
        xData = np.hstack([Data[:np.shape(self.data)[0] - 1, 0 : 4], Data[:np.shape(self.data)[0] - 1, 5].reshape((np.shape(self.data)[0] - 1,1))])
        yData = Data[1:np.shape(self.data)[0], 0 : 4]
        return xData, yData, [SL_max, SL_min, SL_avg],[SM_max,SM_min,SM_avg],[SR_max,SR_min,SR_avg],[Dis_max, Dis_min, Dis_avg], [Act_max, Act_min, Act_avg]
    #绘图方法
    def initPlot(self, traiName):
        self.fig = plt.figure(1, figsize=(24, 12), dpi=80)
        self.SensorL = self.fig.add_subplot(2, 3, 1)
        self.SensorL.set_title('%sSensorL'%traiName)
        plt.xlabel('Iteration')
        plt.ylabel('Result')
        self.SensorM = self.fig.add_subplot(2, 3, 2)
        self.SensorM.set_title('%sSensorM'%traiName)
        plt.xlabel('Iteration')
        plt.ylabel('Result')
        self.SensorR = self.fig.add_subplot(2, 3, 3)
        self.SensorR.set_title('%sSensorR'%traiName)
        plt.xlabel('Iteration')
        plt.ylabel('Result')
        self.Distance = self.fig.add_subplot(2, 3, 4)
        self.Distance.set_title('%sDistance'%traiName)
        plt.xlabel('Iteration')
        plt.ylabel('Distance')
        # self.Reward = self.fig.add_subplot(3, 3, 5)
        # self.Reward.set_title('%sReward'%traiName)
        # plt.xlabel('Iteration')
        # plt.ylabel('Reward')
        self.finalLoss = self.fig.add_subplot(2, 3, 5)
        self.finalLoss.set_title('%sFinalLoss' % traiName)
        plt.xlabel('Episode')
        plt.ylabel('FinalLoss')
        self.Loss = self.fig.add_subplot(2, 3, 6)
        self.Loss.set_title('%sLoss'%traiName)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.ion()  # 使图标实时显示
        plt.show()

    def run(self):
        Net = NN_Net(5, 4, activationFun=tf.nn.relu, netName='NN_Net')
        self.episode = 1
        self.lossList = []
        self.FinalLossList = []
        xData, yData, SL, SM, SR, Dis, Act = self.reshapData()
        while True : #一次训练周期结束后再次挑选数据开始训练

            #print('X:',xData)
            #print('Y:',yData)
            # file = open('log/%s%slog/%s_episode%s_xData.txt'%(Net.TIME, Net.netName,self.traiName, self.episode), 'w')
            # file.write(str(xData))
            # file.close()
            # file = open('log/%s%slog/%s_episode%s_yData.txt'%(Net.TIME, Net.netName,self.traiName, self.episode), 'w')
            # file.write(str(yData))
            # file.close()
            self.initPlot('%s_episode_%s'%(self.traiName,self.episode))
            self.SensorL.scatter(np.arange(np.shape(self.data)[0] - 1), (yData[:, 0]*(SL[0] - SL[1]) + SL[2]), c = 'b', s = 5, marker = 'o')
            self.SensorM.scatter(np.arange(np.shape(self.data)[0] - 1), (yData[:, 1]*(SM[0] - SM[1]) + SM[2]), c='b', s=5, marker='o')
            self.SensorR.scatter(np.arange(np.shape(self.data)[0] - 1), (yData[:, 2]*(SR[0] - SR[1]) + SR[2]), c='b', s=5, marker='o')
            self.Distance.scatter(np.arange(np.shape(self.data)[0] - 1), (yData[:, 3]*(Dis[0] - Dis[1]) + Dis[2]), c='b', s=5, marker='o')
            # self.Reward.scatter(np.arange(100), (yData[:, 4]*(Rew[0] - Rew[1]) + Rew[2]), c='b', s=5, marker='o')
            self.finalLoss.plot(np.arange(len(self.FinalLossList)), self.FinalLossList, 'r-', lw=1)
            #self.Loss.plot(self.ITE, self.lossList, 'r-', lw = 1)
            Net.episode = self.episode
            self.push = False
            while not self.push:#持续训练，直到loss满足一定条件
                self.Iteration += 1
                Net.train(xData, yData)
                loss = Net.sess.run(Net.loss, feed_dict={Net.xDataPH: xData, Net.yDataPH: yData})

                if self.Iteration % 4000 == 0:
                    self.push = True

                if self.Iteration % 50 == 0:
                    self.trainResult = Net.sess.run(Net.mergeOP, feed_dict={Net.xDataPH:xData, Net.yDataPH:yData})
                    Net.writer.add_summary(self.trainResult, self.Iteration)
                    self.lossList.append(loss)
                    self.ITE.append(self.Iteration)
                    plt.cla()
                    self.Loss.plot(self.ITE, self.lossList, 'r-', lw=1)
                    plt.text(np.array(self.ITE).max()*0.8, np.array(self.lossList).max(), 'Loss = %.4f' % loss, fontdict={'size': 10, 'color': 'red'}, ha = 'center', va = 'center')
                    Net.saver.save(Net.sess, 'log/%s%slog/episode%s_model/' % (Net.TIME, Net.netName,self.episode), global_step=self.Iteration)
                    print(('Epi%s_Ite%s_loss:  %s'%(self.episode, self.Iteration, loss)))
                    plt.pause(0.01)
                #print(loss)
                #last_loss = loss
            #每个训练周期结束后，显示训练结果
            self.FinalLossList.append(loss)
            self.predictValue = Net.sess.run(Net.output, feed_dict={Net.xDataPH:xData})
            self.SensorL.scatter(np.arange(np.shape(self.data)[0] - 1), (self.predictValue[:, 0]*(SL[0] - SL[1]) + SL[2]), c='orange', s=3, marker='+')
            self.SensorM.scatter(np.arange(np.shape(self.data)[0] - 1), (self.predictValue[:, 1]*(SM[0] - SM[1]) + SM[2]), c='orange', s=3, marker='+')
            self.SensorR.scatter(np.arange(np.shape(self.data)[0] - 1), (self.predictValue[:, 2]*(SR[0] - SR[1]) + SR[2]), c='orange', s=3, marker='+')
            self.Distance.plot(np.arange(np.shape(self.data)[0] - 1), (self.predictValue[:, 3]*(Dis[0] - Dis[1]) + Dis[2]), 'r-', lw = 1)
            # self.Reward.plot(np.arange(100), (self.predictValue[:, 4]*(Rew[0] - Rew[1]) + Rew[2]), 'r-', lw = 1)
            plt.savefig('log/%s%slog/%s_sepisode_%s.png'%(Net.TIME, Net.netName, self.traiName, self.episode))
            plt.ioff()
            plt.close()
            self.episode += 1

#NN网络类
class NN_Net():
    def __init__(self, inPutSize = 1, outPutSize = 1, activationFun = None, netName = None):
        self.TIME = time.strftime('%Y_%m_%d_%H_%M_%S_', time.localtime(time.time()))
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        self.inPutSize = inPutSize
        self.outPutSize = outPutSize
        self.netName = netName
        self.activationFun = activationFun
        self.episode = 0
        self.buildNet()
        #self.loss = tf.losses.mean_squared_error(labels=self.yDataPH, predictions=self.output)
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.yDataPH - self.output), reduction_indices = 1))
        tf.summary.scalar('EPI%s_loss'%self.episode, self.loss)
        self.trainStep = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999,
                                        epsilon=1e-08).minimize(self.loss)
        self.init = tf.initialize_all_variables()
        self.saver = tf.train.Saver(max_to_keep = 4)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options))
        self.sess.run(self.init)
        self.writer = tf.summary.FileWriter('log/%s%slog/'%(self.TIME,self.netName),self.sess.graph)
        self.mergeOP = tf.summary.merge_all()

    #添加层
    def addLayer(self, inputs, inPutSize, outPutSize, layername):
        with tf.name_scope(layername):
            Weights = tf.Variable(tf.random_normal([inPutSize, outPutSize]), name='weight')
            tf.summary.histogram('%s_Weight' % layername, Weights)
            bias = tf.Variable(tf.zeros([1, outPutSize]) + 0.1, name='bias')
            tf.summary.histogram('%s_bias' % layername, bias)
            WxPlusBias = tf.matmul(inputs, Weights) + bias
            if self.activationFun is None:
                output = WxPlusBias
            else:
                output = self.activationFun(WxPlusBias)
            tf.summary.histogram('%s_output' % layername, output)
            return output

    #数据空间
    def buildDataPH(self):
        with tf.name_scope('inputs'):
            self.xDataPH = tf.placeholder(tf.float32, [None, 5], name='X_INPUT')
            self.yDataPH = tf.placeholder(tf.float32, [None, 4], name='Y_INPUT')

    #网络生成
    def buildNet(self):
        self.buildDataPH()
        self.inputLayer = self.addLayer(self.xDataPH, self.inPutSize, 20, 'input_layer' )
        self.hideLayer_1 = self.addLayer(self.inputLayer, 20, 100, 'hideLayer_1')
        self.hideLayer_2 = self.addLayer(self.hideLayer_1, 100, 20, 'hideLayer_3')
        with tf.name_scope('outputLayer'):
            Weights = tf.Variable(tf.random_normal([20, self.outPutSize]), name='weight')
            tf.summary.histogram('outputLayer_Weight', Weights)
            bias = tf.Variable(tf.zeros([1, self.outPutSize]) + 0.1, name='bias')
            tf.summary.histogram('outputLayer_bias', bias)
            self.output = tf.matmul(self.hideLayer_2, Weights) + bias


    def train(self, xData, yData):
        self.sess.run(self.trainStep, feed_dict={self.xDataPH:xData, self.yDataPH:yData})



if __name__ == '__main__':
    Train_OP = trainOP('PCRecord.txt', 'Evn_Net_NN')
    Train_OP.run()