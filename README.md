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
        
 
