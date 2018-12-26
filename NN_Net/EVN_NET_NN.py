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
    #随机函数
    def random(self, a, b):
        return random.randint(a, b)

    #导入数据
    def loadData(self, filename):
        return np.loadtxt(filename)

    #挑选数据
    def pickData(self):
        a = self.random(0, (np.shape(self.data)[0] - 101))
        return self.data[a:a+101, :]

    def reshapData(self):
        return self.pickData()[0:100, 0 : 6], self.pickData()[1:101, 0:5]
    #绘图方法
    def initPlot(self, traiName):
        self.fig = plt.figure(1, figsize=(24, 8), dpi=80)
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
        self.Reward = self.fig.add_subplot(2, 3, 5)
        self.Reward.set_title('%sReward'%traiName)
        plt.xlabel('Iteration')
        plt.ylabel('Reward')
        self.Loss = self.fig.add_subplot(2, 3, 6)
        self.Loss.set_title('%sLoss'%traiName)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.ion()  # 使图标实时显示
        plt.show()

    def run(self):                  
        Net = NN_Net(6, 5, activationFun=tf.nn.relu, netName='NN_Net')
        self.episode = 1
        self.lossList = []
        while True :
            xData, yData = self.reshapData()
            print('X:',xData)
            print('Y:',yData)
            self.initPlot('%s_episode_%s'%(self.traiName,self.episode))
            self.SensorL.scatter(np.arange(100), yData[:, 0], c = 'b', s = 3, marker = 'o')
            self.SensorM.scatter(np.arange(100), yData[:, 1], c='b', s=3, marker='o')
            self.SensorR.scatter(np.arange(100), yData[:, 2], c='b', s=3, marker='o')
            self.Distance.scatter(np.arange(100), yData[:, 3], c='b', s=3, marker='o')
            self.Reward.scatter(np.arange(100), yData[:, 4], c='b', s=3, marker='o')
            self.Loss.plot(np.arange(len(self.lossList)), self.lossList, 'r-', lw = 3)
            Net.episode = self.episode
            self.push = False
            while not self.push:
                self.Iteration += 1
                Net.train(xData, yData)
                # try :
                #     if last_loss - loss < 0.01:
                #         self.push = False
                # except Exception:
                #     pass
                loss = Net.sess.run(Net.loss, feed_dict={Net.xDataPH: xData, Net.yDataPH: yData})
                # if self.Iteration % 50 == 0:
                #     self.trainResult = Net.sess.run(Net.mergeOP, feed_dict={Net.xDataPH:xData, Net.yDataPH:yData})
                #     Net.writer.add_summary(self.trainResult, self.Iteration)
                #     self.lossList.append(loss)
                #     plt.cla()
                #     self.Loss.plot(np.arange(len(self.lossList)), self.lossList, 'r-', lw=3)
                #     plt.text(0.5, 0, 'Loss = %.4f' % loss, fontdict={'size': 10, 'color': 'red'})
                #     print(('%s:%s'%(self.Iteration,self.lossList)))
                #     plt.pause(0.5)
                print(loss)
                last_loss = loss
            self.lossList = []
            self.predictValue = Net.sess.run(Net.output, feed_dict={Net.xDataPH:xData})
            self.SensorL.scatter(np.arange(100), self.predictValue[:, 0], c='r', s=1, marker='.')
            self.SensorM.scatter(np.arange(100), self.predictValue[:, 1], c='r', s=1, marker='.')
            self.SensorR.scatter(np.arange(100), self.predictValue[:, 2], c='r', s=1, marker='.')
            self.Distance.plot(np.arange(100), self.predictValue[:, 3], 'r-', lw = 1)
            self.Reward.plot(np.arange(100), self.predictValue[:, 4], 'r-', lw = 1)
            plt.savefig('%s_sepisode_%s.png'%(self.traiName, self.episode))
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
        self.loss = tf.losses.mean_squared_error(labels=self.yDataPH, predictions=self.output)
        #self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.yDataPH - self.output), reduction_indices = 1))
        tf.summary.scalar('EPI%s_loss'%self.episode, self.loss)
        self.trainStep = tf.train.AdamOptimizer(0.1).minimize(self.loss)
        self.init = tf.initialize_all_variables()
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options))
        self.sess.run(self.init)
        self.writer = tf.summary.FileWriter('%s%slog/'%(self.TIME,self.netName),self.sess.graph)
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
            self.xDataPH = tf.placeholder(tf.float32, [None, 6], name='X_INPUT')
            self.yDataPH = tf.placeholder(tf.float32, [None, 5], name='Y_INPUT')

    #网络生成
    def buildNet(self):
        self.buildDataPH()
        self.inputLayer = self.addLayer(self.xDataPH, self.inPutSize, 16, 'input_layer' )
        self.hideLayer_1 = self.addLayer(self.inputLayer, 16, 16, 'hideLayer_1')
        self.hideLayer_2 = self.addLayer(self.hideLayer_1, 16, 8, 'hideLayer_2')
        self.output = self.addLayer(self.hideLayer_2, 8, self.outPutSize, 'output_Layer')

    def train(self, xData, yData):
        self.sess.run(self.trainStep, feed_dict={self.xDataPH:xData, self.yDataPH:yData})


if __name__ == '__main__':
    Train_OP = trainOP('PCRecord.txt', 'Evn_Net_NN')
    Train_OP.run()