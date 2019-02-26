import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random


#随机函数
def random (a , b):
    return random.randint(a, b)

#导入数据
def loadData(filename):
    return np.loadtxt(filename)

def pickData():

    random(0, 547)

#定义添加层函数
def addLayer (inputs, inputSize, outputSize, activationFunction = None, layername = None):
    with tf.name_scope(layername):
        Weights = tf.Variable(tf.random_normal([inputSize, outputSize]), name = 'weight')
        tf.summary.histogram('%s_Weight' % layername, Weights)
        bias = tf.Variable(tf.zeros([1, outputSize]) + 0.1, name = 'bias')
        tf.summary.histogram('%s_bias' % layername, bias)
        WxPlusBias = tf.matmul(inputs, Weights) + bias
        if activationFunction is None :
            output = WxPlusBias
        else:
            output = activationFunction(WxPlusBias)
        tf.summary.histogram('%s_output'%layername, output)
        return output

#tf_GPU 使用权限配置
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.7)

#数据生成

xData = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, xData.shape)#噪点
yData = np.square(xData) - 0.5 + noise

#定义数据存放空间
with tf.name_scope ('inputs'):
    xs = tf.placeholder(tf.float32, [None,1], name = 'x_input')
    ys = tf.placeholder(tf.float32, [None,1], name = 'y_input')

#输入层
l1 = addLayer(xs, 1, 10, activationFunction = tf.nn.relu, layername='input_L')

#输出层
prediction = addLayer(l1, 10, 1, activationFunction = None, layername='output_L')

#loss函数
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices = 1))
    tf.summary.scalar('loss', loss)
#定义train step， init variable, session
with tf.name_scope('train'):
    trainStep = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
sess.run(init)

#日志
writer = tf.summary.FileWriter('log/', sess.graph)
merge_op = tf.summary.merge_all()
#数据表动态显示
fig = plt.figure(1, figsize = (10, 8), dpi = 120)
ax = fig.add_subplot(1, 1, 1)
ax.scatter(xData, yData)
plt.ion()#使图标实时显示
plt.show()
plt.xlabel('x')
plt.ylabel('y')
plt.title('regression')

#训练开始
for i in range (1000):
    sess.run(trainStep, feed_dict={xs: xData, ys: yData})
    if i%50 == 0:
        #print(sess.run(loss, feed_dict={xs:xData, ys:yData}))
        result = sess.run(merge_op, feed_dict={xs:xData, ys:yData})
        writer.add_summary(result, i)
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        predictionValue = sess.run(prediction, feed_dict = {xs: xData})
        lines = ax.plot(xData, predictionValue, 'r-', lw = 3)
        print(sess.run(loss, feed_dict={xs: xData, ys: yData}))
        plt.pause(0.05)

#数据表保存
plt.savefig("regression.png")
plt.ioff()
