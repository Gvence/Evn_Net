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