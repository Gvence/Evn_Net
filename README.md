# Evn_Net
# NN_Net_19_12_26: Net_Structure(5 - 10 - 20 - 20 - 10 - 5 )

        Input:S_L S_M S_R Dis Act
        Output:S_L S_M S_R Dis Rew
        Need Data normalization loss Error, no GradientDescent
        Result like log 2019_01_03_20_06_06_NN_Netlog

# NN_Net_19_1_04: Net_Structure(5 - 20 - 100 - 20 -5)

        Input: S_L S_M S_R Dis Act
        Output: S_L S_M S_R Dis 
        Can good fitting 
        6000 samples every times
        some sensor result will be ignored because its sample size is very small

# Car_Remote_Control_19_1_11:
        
        Pi_Car remote control code
        U can use this code to control the pi_Car with terminal in pi
        U should connect with pi through VNCViewer first

# SensorDataRecord_19_1_11:
        
        Sensor Result of remote control car
        Sensor data include: S_L S_M S_R DIS1 DIS2 ACT
 
# NN_Net_19_1_11: Net_Structure(6 - 20 - 50 - 20 - 5 )
        
        Trainning code of sensor data
        Input the whole sensor data every episode:S_L S_M S_R DIS1 DIS2 ACT
        Output :S_L S_M S_R DIS1 DIS2
        Env is a statics state, but in this method of trainning it think of the enviroment
                as some linear states
                
# RNN_Net_19_2_26: Net_Structure(6 - 512 - 5 )

        record_ata.txt from map.py 
        Input :S_L S_M S_R DIS1 DIS2 ACT
        Output :S_L S_M S_R DIS1 DIS2
        Rnn 具有时序性，尝试使用RNN
        现实场景数据采集受限，在虚拟场景采数据
        加入了数据归一化Batch Normalization
        数据量十分庞大需要数据预处理，制作训练集，测试级，验证集用于后期网络性能验证（参考图像数据集制作）
        需要增加网络模型保存方法

# RNN_Net_19_2_27: Net_Structure(6 - 512 - 5)
	
        Input :S_L S_M S_R DIS1 DIS2 ACT
        Output :S_L S_M S_R DIS1 DIS2
        Data_Processing.py用于数据集制作
        完成数据集制作train 0.7 test 0.15 val 0.15
        完成模型保存和重加载，网络可以进行间歇训练
        需要增加交叉验证方法
        了解网络中交叉熵加softmax的loss定义方法

# RNN_Net_19_2_28: Net_Structure(6 - 512 - 5)

        Input :S_L S_M S_R DIS1 DIS2 ACT
        Output :S_L S_M S_R DIS1 DIS2
        将TimeStep从5>>1， 也就是没做出一次动作结果预测反传一次误差，这样有利于网络描述单个动作的反馈结果的准确性，而之前每五个动作后再次反传误差，不利于单个动作预测，而趋向于连续动作做出后的结果预测，这与正常的环境反馈是存在差异的，正常的环境反馈是在一个动作执行后得到的，而不是一连串动作后才有反馈，每个动作都有与之对应的环境反馈
        将BatchSize从10>>500 增加一个Batch 中样本的覆盖范围，使模型在一个iteration内能够更好的提取环境的总体特征
        完成了train_set, validation_set, test_set的制作
        train_set:通过训练和误差传递，修正网络参数(W, B),建立一个数据到标签的抽象关系——MODEL
        validation_set:对学习出来的MODEL调整超级参数（网络隐藏层数， 网络单元数， 训练次数等）
        test_set:验证MODEL的准确性

# RNN_Net_19_03_01: Net_Structure(6 - 512 - 5)

        Input :S_L S_M S_R DIS1 DIS2 ACT
        Output :S_L S_M S_R DIS1 DIS2
        将Batch_Start的选取从连续选取变成随机选取
        train_set,val_set,test_set 分开三次采集数据

# RNN_Net_19_03_02: Net_Structure(6 - 512 - 3)

        Input :S_L S_M S_R DIS1 DIS2 ACT
        Output :S_L S_M S_R
        在网络的输入层加入了relu激活函数，发现效果明显改善，relu对于0 1 的二分类预测非常有效
        将红外传感器的0 1数据与距离传感器的距离数据分开训练，隔绝距离大误差对传感器小误差的干扰
        通过Val_loss 和 train_loss曲线的对比可以发现，512个RNN cell units 存在过拟合现象
        消除过拟合方法：
                1：在训练数据中加入噪声（高斯， 正态分布。。）
                2：减少RNN cell units（通过交叉验证曲线逐步找到最佳cell units）

# RNN_Net_19_03_03: Net_Structure(3 - 512 - 2)
        
        Input :DIS1 DIS2 ACT
        Output :DIS1 DIS2
        距离的预测还是存在较大误差，已经将relu去掉，直接用全连接层左输入
        提高了学习率0.006>>0.1
        距离数据的交叉验证同样存在过拟合问题，验证集loss很高
        



 
# RNN_Net_19_03_03: rnn_env_dis.py(3 - 64 - 2)
		   rnn_env_sensor.py(6 - 64 - 3)
	

        在sensor的训练中开启了normalize， dis训练中未开启。
        训练效果大幅提升。
        紧完成了sensor的单步的预测，但是每次要预测下一个状态都需要输入之前的500个连续状态
        尝试缩短需要的序列长度，看能缩减的最小数值。
        dis的测试没有做，预测应该是类似的
        将sensor与dis混合，让程序同事预测dis和sensor
        可能涉及到用tf制作两个sess，然后分别加载两个模型同事运行，不确定能否实现，有待验证



# RNN_Net_19_03_13: rnn_env_dis.py(3 - 64 - 2)
		   rnn_env_sensor.py(6 - 64 - 3)

        mixed_processing,成功将距离数据和sensor数据模型混合调用
        考虑将Env和reforceLearning结合
        经验发现64个cell比128个cell的RNN更好收敛，而且预测效果很好，可以继续调至32cell试试
        模型调用很慢


						                          6 - 64 - 3
# RNN_Net_19_03_14: mixed_train_processing.py (6 -            - 5)
						                          3 - 64 - 2	

        mixed_train_processing.py成功将网络变成Dis 和Sensor 分流训练
        创建了双RNN核，用于两个数据流的训练
        需要等待训练结果，如果非常好就开始结合强化学习

						                          6 - 32 - 3
# RNN_Net_19_03_25: mixed_train_processing.py (6 -            - 5)
						                          3 - 32 - 2	

        随机选取BATCH_START,然后进行连续性训练
        每次都对时间片段的下一个状态进行预测，然后将时间片段向前推进一个时间点
        存在疑惑：
            所有的BATCH的时间都是连续的，虽然根据BATCH_SIZE选取了多个时间片段
            但是这些时间片段都是连载一起的，训练效果存在质疑
            打算根据BATCH_SIZE随机选取多个时间片段，再进行时间点延伸训练


						                          6 - 32 - 3
# RNN_Net_19_04_02: mixed_train_processing.py (6 -            - 5)
						                          3 - 32 - 2	
        编写了一套新的数据获取和训练框架
        首先随机选取BATCH_SIZE个BATCH_START（时间片段起始点）
        每次训练结束，所有的BATCH的时间点加一（即向前延伸一个时间点）
        当延伸STEP_LENGTH个时间点后，一个训练回合结束（iteration+1）
        当训练的数据量大致等于数据的总量时即完成一个训练周期（EPO + 1）
        适当的增加BATCH_SIZE可以提升训练效率，加快网络的收敛
        多个BATCH_SIZE作为一个tensor输入网络可以使网络更好把握收敛方向，提高效率和成功率

# RNN_NET_19_04_17:
# map.py (2019_04_03):
    1.完成class MyPaintWidget()的修改
    2.实现了记录绘制的障碍线并可以根据记录文件自动恢复之前记录的线
    3.将决策网络改成了基于TF的Dqn
    4.class game() -> update() ：
        --加入brain.choose_action()用于根据当前状态输出动作
        --当根据生层的action移动后，新的动作作为下一个状态
        --使用brain.store_transition()保存当前状态，动作，回报,下一个状态

# RL_brain.py (2019_04_04):
    1.class DeepQNetwork() -> store_transition():
        --每当存够一个size,用memory_management()将数据保存
    2.class DeepQNetwork():
        --加入memory_management()方法
        --加入read_memorystack(),可以直接返回保存好的stack

# mixed_train_processing.py(2019_04_08):
    1.calss LSTMRNN -> init_model():
        --将所有的网路参数放入LSTMRNN类中，并给出了默认值，可在实例化时修改
        --初始化模型，检查是否已经存在模型，如果存在可以根据调整iteration使
            模型接着上个iteration训练，如果iteration设置0，则重新训练
        --实例化sess,saver,writer,merged.定义gpu_option
        
# map.py(2019_04_08):
    1.class CarApp() -> train_EvnNet():
        --在App中加如环境网络训练，每秒调用一定次数
        --成功完成App下调用dqn -> read_memorystack()获取数据用于LSTMRNN训练
        
# mixed_train_processing.py(2019_04_09):
    1.class LSTMRNN -> init_model():
        --加入了两个对象属性INITSTATE,times_tip。
        --INITSTATE用于判断何时初始化batch_start
        --times_tip用于判断总的训练次数
    2.class LSTMRNN -> get_batch():
        --用于从trainData中解读出训练网络需要的input和target
        --dqn的memory中，状态是连续存放的，其存放结构是：
            memory: array[
                          [stateRightNow, action, reward, stateNetTime]
                         ]
                    memory.shape : (-1, 12)
                    state**.shape : (5)  ## it include s1 s2 s3 dis1 dis2
                    action.shape : (1) ## 0 1 2  in rotation[0, 20, -20]
                    reward.shape : (1)  
    3.class LSTMRNN -> train():
        --当训练进行时,TRAINING = True,防止App在网络正在训练过程中又一次调取训练函数
        --当一个训练回合结束时，需要重新抽取数据，此时将INITSTATE = True,这样当app再次调取训练
            函数时，就会重新选取一个memory,并初始化batch_start

# map.py(2019_04_10):
    1.class LSTMRNN -> plotting():
        --调取Evn中的记录list，并绘制
        --每次训练，保存后20%的batch的label和pred
        --一个iteration结束后，绘制一次数据图，数据量为step_length * 0.2*batch_size
        
# mixed_train_processing.py(2019_04_13):
    1. class LSTMRNN -> train():
        --每完成一个memory的训练，保存一次模型，记录一次loag
    
# RL_brain.py(2019_04_14):
    1.class DeepQNetwork -> store_transition():
        --将真实环境和模型环境生成的数据的保存操作放在一起。区别在与模型环境只有一个memory,里面存放了
            memory size个状态的信息，而真实环境的数据保存后，会放入stack，等stack存满后写入到txt文件
            
# map.py (2019_04_14):
    1.class CarApp -> train_DQN():
        --从memory中随机选一个状态S作为dqn网络的初始状态输入
        --dqn返回一个动作A
        --将S,A输入到Evn_net得到一个S_
        --根据S_认为给出一个R
        --将S, A, R, S_ 作为一个memory存入M_Evn_memory
        --调用dqn -> learn ,根据M_Evn_memory训练dqn:
            (Q_eval - R + gamma*(q(s_).max))^2






















