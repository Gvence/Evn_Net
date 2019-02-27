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
 
