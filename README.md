# Evn_Net
NN_Net_19_12_26: Net_Structure(5 - 10 - 20 - 20 - 10 - 5 )

        Input:S_L S_M S_R Dis Act
        Output:S_L S_M S_R Dis Rew
        Need Data normalization loss Error, no GradientDescent
        Result like log 2019_01_03_20_06_06_NN_Netlog
NN_Net_19_1_04: Net_Structure(5 - 20 - 100 - 20 -5)

        Input: S_L S_M S_R Dis Act
        Output: S_L S_M S_R Dis 
        Can good fitting 
        6000 samples every times
        some sensor result will be ignored because its sample size is very small

Car_Remote_Control_19_1_11:
        
        Pi_Car remote control code
        U can use this code to control the pi_Car with terminal in pi
        U should connect with pi through VNCViewer first

SensorDataRecord_19_1_11:
        
        Sensor Result of remote control car
        Sensor data include: S_L S_M S_R DIS1 DIS2 ACT
 
NN_Net_19_1_11: Net_Structure(6 - 20 - 50 - 20 - 5 )
        
        Trainning code of sensor data
        Input the whole sensor data every episode:S_L S_M S_R DIS1 DIS2 ACT
        Output :S_L S_M S_R DIS1 DIS2
        Env is a statics state, but in this method of trainning it think of the enviroment
                as some linear states
                
RNN_Net_19_2_26: Net_Structure(6 - 512 - 5 )

        record_ata.txt from map.py 
        Input :S_L S_M S_R DIS1 DIS2 ACT
        Output :S_L S_M S_R DIS1 DIS2
        Rnn 具有时序性，尝试使用RNN
        现实场景数据采集受限，在虚拟场景采数据
        加入了数据归一化Batch Normalization
        数据量十分庞大需要数据预处理，制作训练集，测试级，验证集用于后期网络性能验证（参考图像数据集制作）
        需要增加网络模型保存方法

RNN_Net_19_2_27: Net_Structure(6 - 512 - 5)
	
        Input :S_L S_M S_R DIS1 DIS2 ACT
        Output :S_L S_M S_R DIS1 DIS2
        Data_Processing.py用于数据集制作
        完成数据集制作train 0.7 test 0.15 val 0.15
        完成模型保存和重加载，网络可以进行间歇训练
        需要增加交叉验证方法
        了解网络中交叉熵加softmax的loss定义方法

 RNN_Net_19_2_28: Net_Structure(6 - 512 - 5)

        Input :S_L S_M S_R DIS1 DIS2 ACT
        Output :S_L S_M S_R DIS1 DIS2
        将TimeStep从5>>1， 也就是没做出一次动作结果预测反传一次误差，这样有利于网络描述单个动作的反馈结果的准确性，而之前每五个动作后再次反传误差，不利于单个动作预测，而趋向于连续动作做出后的结果预测，这与正常的环境反馈是存在差异的，正常的环境反馈是在一个动作执行后得到的，而不是一连串动作后才有反馈，每个动作都有与之对应的环境反馈
        将BatchSize从10>>500 增加一个Batch 中样本的覆盖范围，使模型在一个iteration内能够更好的提取环境的总体特征
        完成了train_set, validation_set, test_set的制作
        train_set:通过训练和误差传递，修正网络参数(W, B),建立一个数据到标签的抽象关系——MODEL
        validation_set:对学习出来的MODEL调整超级参数（网络隐藏层数， 网络单元数， 训练次数等）
        test_set:验证MODEL的准确性

 RNN_Net_19_03_01: Net_Structure(6 - 512 - 5)

        Input :S_L S_M S_R DIS1 DIS2 ACT
        Output :S_L S_M S_R DIS1 DIS2
        将Batch_Start的选取从连续选取变成随机选取
        train_set,val_set,test_set 分开三次采集数据

 RNN_Net_19_03_02: Net_Structure(6 - 512 - 3)

        Input :S_L S_M S_R DIS1 DIS2 ACT
        Output :S_L S_M S_R
        在网络的输入层加入了relu激活函数，发现效果明显改善，relu对于0 1 的二分类预测非常有效
        将红外传感器的0 1数据与距离传感器的距离数据分开训练，隔绝距离大误差对传感器小误差的干扰
        通过Val_loss 和 train_loss曲线的对比可以发现，512个RNN cell units 存在过拟合现象
        消除过拟合方法：
                1：在训练数据中加入噪声（高斯， 正态分布。。）
                2：减少RNN cell units（通过交叉验证曲线逐步找到最佳cell units）

 RNN_Net_19_03_03: Net_Structure(3 - 512 - 2)
        
        Input :DIS1 DIS2 ACT
        Output :DIS1 DIS2
        距离的预测还是存在较大误差，已经将relu去掉，直接用全连接层左输入
        提高了学习率0.006>>0.1
        距离数据的交叉验证同样存在过拟合问题，验证集loss很高
        





































