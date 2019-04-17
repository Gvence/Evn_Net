import signal
import time
from mixed_train_processing import LSTMRNN
import random
import numpy as np
from RL_brain import DeepQNetwork
# Define signal handler function
def myHandler(signum, frame):
    print("TimeoutError")



class test ():
    def __init__(self):
        #np.random.seed(1)
        # RNN = LSTMRNN( )
        # dqn = DeepQNetwork()
        # dic = dqn.read_memorystack()
        # trainData = dic[str(randint(0, len(dic) - 1))]
        # RNN.get_batch(trainData)
        # RNN.train()
        a = np.array([1,2,3,4,5,6])
        b = 0
        while b!= 10:
            b = np.random.randint(0, 10)
            print(b)





if __name__ == '__main__':
    T = test()