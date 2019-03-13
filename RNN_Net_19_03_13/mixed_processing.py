import numpy as np
from multiprocessing import Process
import tensorflow as tf
tf.reset_default_graph()
import matplotlib.pyplot  as plt
from rnn_env_sensor import LSTMRNN_S
from rnn_env_dis import LSTMRNN_D
from Data_Processing import dataRebuild
import random

#加载距离数据，并重构
def get_D_Data(data):
    batch_start = random.randint(0, len(data) - 1 - D_TIME_STEPS*D_BATCH_SIZE)
    VTxin = data[batch_start:batch_start + D_TIME_STEPS*D_BATCH_SIZE , [3, 4, 5]].reshape((-1,D_TIME_STEPS ,D_INPUT_SIZE))
    VTlabel = data[batch_start + 1 : batch_start + D_TIME_STEPS * D_BATCH_SIZE + 1, [3, 4]].reshape((-1, D_TIME_STEPS, D_OUTPUT_SIZE))
    return [VTxin, VTlabel]

#加载传感器数据，并重构
def get_S_Data(data):
    batch_start = random.randint(0, len(data) - 1 - S_TIME_STEPS*S_BATCH_SIZE)
    VTxin = data[batch_start:batch_start + S_TIME_STEPS*S_BATCH_SIZE , :].reshape((-1,S_TIME_STEPS ,S_INPUT_SIZE))
    VTlabel = data[batch_start + 1 : batch_start + S_TIME_STEPS * S_BATCH_SIZE + 1, [0, 1, 2]].reshape((-1, S_TIME_STEPS, S_OUTPUT_SIZE))
    return [VTxin, VTlabel]
if __name__ == '__main__':
    D_pred_list = []
    D_l_list = []
    S_pred_list = []
    S_l_list = []
    plt.ion()  # 设置连续 plot
    plt.show()
    fig = plt.figure(1, figsize=(24, 12), dpi=80)
    # 加载数据
    buildData = dataRebuild('Data.txt', 0.7, 0.15)
    trainData = np.loadtxt('trainData.txt')
    testData = np.loadtxt('testData.txt')
    valData = np.loadtxt('valData.txt')
    D_TIME_STEPS = 1  # backpropagation through time 的time_steps 误差反传的步长
    D_BATCH_SIZE = 500  # 每次训练的步长的数量
    D_INPUT_SIZE = 3  # 数据输入size
    D_OUTPUT_SIZE = 2  # 数据输出 size
    D_CELL_SIZE = 64
    D_lr = 0.006
    D_MODEL_NAME = 'iteration_model-1325000'
    D_model = LSTMRNN_D(D_TIME_STEPS, D_INPUT_SIZE, D_OUTPUT_SIZE, D_CELL_SIZE, D_BATCH_SIZE, D_lr, norm=False)
    graphD = tf.get_default_graph()
    sessD = tf.Session(graph=graphD)
    iteration = 0
    with sessD.as_default():
        with graphD.as_default():
            sessD.run(tf.global_variables_initializer())
            saverD = tf.train.Saver()
            saverD.restore(sessD, 'dis_model/%s'%D_MODEL_NAME)
            while iteration <= 100:
                iteration += 1
                D_x, D_l = get_D_Data(valData)
                D_l_list.append(D_l.reshape(-1, D_OUTPUT_SIZE)[-1, :])
                D_pred = sessD.run(D_model.pred, feed_dict={D_model.xd: D_x})
                D_pred_list.append(D_pred.reshape(-1, D_OUTPUT_SIZE)[-1, :])
    sessD.close()
    tf.reset_default_graph()
    S_TIME_STEPS = 1  # backpropagation through time 的time_steps 误差反传的步长
    S_BATCH_SIZE = 500  # 每次训练的步长的数量
    S_INPUT_SIZE = 6  # 数据输入size
    S_OUTPUT_SIZE = 3  # 数据输出 size
    S_CELL_SIZE = 64  # RNN的 hidden unit size
    S_LR = 0.006  # learning rate
    S_MODEL_NAME = 'iteration_model-4490000'
    S_model = LSTMRNN_S(S_TIME_STEPS, S_INPUT_SIZE, S_OUTPUT_SIZE, S_CELL_SIZE, S_BATCH_SIZE, S_LR, norm=True)
    graphS = tf.get_default_graph()
    sessS = tf.Session(graph=graphS)
    iteration = 0
    with sessS.as_default():
        with graphS.as_default():
            sessS.run(tf.global_variables_initializer())
            saverS = tf.train.Saver()
            saverS.restore(sessS, 'sensor_model/%s'%S_MODEL_NAME)
            while iteration <= 100:
                iteration += 1
                S_x, S_l = get_S_Data(valData)
                S_l_list.append(S_l.reshape(-1, S_OUTPUT_SIZE)[-1, :])
                S_pred = sessS.run(S_model.pred, feed_dict={S_model.xs: S_x})
                S_pred_list.append(S_pred.reshape(-1, S_OUTPUT_SIZE)[-1, :])
    sessS.close()


    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


    S_l_list = np.array(S_l_list)
    S_pred_list = np.array(S_pred_list)
    D_l_list = np.array(D_l_list)
    D_pred_list = np.array(D_pred_list)
    chart_1 = fig.add_subplot(5, 1, 1, xlim=(0, S_pred_list.shape[0]))
    chart_1.set_title('valS_L')
    chart_2 = fig.add_subplot(5, 1, 2, xlim=(0, S_pred_list.shape[0]))
    chart_2.set_title('valS_M')
    chart_3 = fig.add_subplot(5, 1, 3, xlim=(0, S_pred_list.shape[0]))
    chart_3.set_title('valS_L')
    chart_4 = fig.add_subplot(5, 1, 4, xlim=(0, D_pred_list.shape[0]))
    chart_4.set_title('Dis_1')
    chart_5 = fig.add_subplot(5, 1, 5, xlim=(0, D_pred_list.shape[0]))
    chart_5.set_title('Dis_2')


    chart_1.scatter(np.arange(0, S_l_list.shape[0]), S_l_list[:, 0], c='blue', s=10, marker='o')
    chart_1.scatter(np.arange(0, S_pred_list.shape[0]), S_pred_list[:, 0], c='red', s=10, marker='o')
    chart_2.scatter(np.arange(0, S_l_list.shape[0]), S_l_list[:, 1], c='blue', s=10, marker='o')
    chart_2.scatter(np.arange(0, S_pred_list.shape[0]), S_pred_list[:, 1], c='red', s=10, marker='o')
    chart_3.scatter(np.arange(0, S_l_list.shape[0]), S_l_list[:, 2], c='blue', s=10, marker='o')
    chart_3.scatter(np.arange(0, S_pred_list.shape[0]), S_pred_list[:, 2], c='red', s=10, marker='o')



    chart_4.plot(np.arange(0, D_l_list.shape[0]), D_l_list[:, 0], 'b-', lw=4)
    chart_4.plot(np.arange(0, D_pred_list.shape[0]), D_pred_list[:, 0], 'r--', lw=2)
    chart_5.plot(np.arange(0, D_l_list.shape[0]), D_l_list[:, 1], 'b-', lw=4)
    chart_5.plot(np.arange(0, D_pred_list.shape[0]), D_pred_list[:, 1], 'r--', lw=2)
    plt.pause(0.1)
    plt.savefig('plot/iteration_%s.png' % iteration)

