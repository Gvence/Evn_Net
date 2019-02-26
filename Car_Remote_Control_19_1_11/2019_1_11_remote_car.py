#coding:utf-8

import os
import threading
import sys
import tty,termios
import time
import datetime
import RPi.GPIO as GPIO
import inspect
import ctypes
import socket
replyData = 0
recivedata = 0
Time = 0
t1 = 0
qt2 = 0
velocity = 0.02
revelocity = 0.03


#############信号引脚定义##############
GPIO.setmode(GPIO.BCM)

#########红外线管脚#############
IR_R = 18
IR_L = 27
IR_M = 22

########电机驱动接口定义#################
ENA = 13	#//L298使能A
ENB = 20	#//L298使能B
IN1 = 19	#//电机接口1
IN2 = 16	#//电机接口2
IN3 = 21	#//电机接口3
IN4 = 26	#//电机接口4

########超声波接口定义#################
ECHO_1 = 4	#超声波1接收脚位
TRIG_1 = 17	#超声波1发射脚位

ECHO_2 = 14	#超声波2接收脚位
TRIG_2 = 15	#超声波2发射脚位
#########管脚类型设置及初始化##########
GPIO.setwarnings(False)
#########电机初始化为LOW##########
GPIO.setup(ENA,GPIO.OUT,initial=GPIO.LOW)
GPIO.setup(ENB,GPIO.OUT,initial=GPIO.LOW)
GPIO.setup(IN1,GPIO.OUT,initial=GPIO.LOW)
GPIO.setup(IN2,GPIO.OUT,initial=GPIO.LOW)
GPIO.setup(IN3,GPIO.OUT,initial=GPIO.LOW)
GPIO.setup(IN4,GPIO.OUT,initial=GPIO.LOW)

GPIO.setup(IR_R,GPIO.IN,pull_up_down=GPIO.PUD_UP)
GPIO.setup(IR_L,GPIO.IN,pull_up_down=GPIO.PUD_UP)
GPIO.setup(IR_M,GPIO.IN,pull_up_down=GPIO.PUD_UP)

##########超声波模块管脚类型设置#########
GPIO.setup(TRIG_1,GPIO.OUT,initial=GPIO.LOW)#超声波模块发射端管脚设置trig
GPIO.setup(ECHO_1,GPIO.IN,pull_up_down=GPIO.PUD_UP)#超声波模块接收端管脚设置echo
GPIO.setup(TRIG_2,GPIO.OUT,initial=GPIO.LOW)
GPIO.setup(ECHO_2,GPIO.IN,pull_up_down=GPIO.PUD_UP)
#########电机电机前进函数##########

run =True
def Motor_TurnRight():
    GPIO.output(ENA,True)
    GPIO.output(ENB,True)
    GPIO.output(IN1,True)
    GPIO.output(IN2,False)
    GPIO.output(IN3,True)
    GPIO.output(IN4,False)
#########电机电机后退函数##########
def Motor_TurnLeft():
    GPIO.output(ENA,True)
    GPIO.output(ENB,True)
    GPIO.output(IN1,False)
    GPIO.output(IN2,True)
    GPIO.output(IN3,False)
    GPIO.output(IN4,True)
#########电机电机左转函数##########
def Motor_Backward():
    GPIO.output(ENA,True)
    GPIO.output(ENB,True)
    GPIO.output(IN1,True)
    GPIO.output(IN2,False)
    GPIO.output(IN3,False)
    GPIO.output(IN4,True)
#########电机电机右转函数##########
def Motor_Forward():
    GPIO.output(ENA,True)
    GPIO.output(ENB,True)
    GPIO.output(IN1,False)
    GPIO.output(IN2,True)
    GPIO.output(IN3,True)
    GPIO.output(IN4,False)
#########电机电机停止函数##########
def Motor_Stop():
    GPIO.output(ENA,True)
    GPIO.output(ENB,True)
    GPIO.output(IN1,False)
    GPIO.output(IN2,False)
    GPIO.output(IN3,False)
    GPIO.output(IN4,False)

####################################################
##函数名称 ：Get_Distence()
##函数功能 超声波测距，返回距离（单位是厘米）
##入口参数 ：无
##出口参数 ：无
####################################################
def	Get_Distance(DisSensor):
    if DisSensor == 1:
        GPIO_TR = TRIG_1
        GPIO_EC = ECHO_1
    if DisSensor == 2:
        GPIO_TR = TRIG_2
        GPIO_EC = ECHO_2
    time.sleep(0.01)
    GPIO.output(GPIO_TR,GPIO.HIGH)
    time.sleep(0.000015)
    GPIO.output(GPIO_TR,GPIO.LOW)
    while not GPIO.input(GPIO_EC):
                pass
    t1 = time.time()
    while GPIO.input(GPIO_EC):
                pass
    t2 = time.time()
    Distence = (t2-t1)*340/2*100
    time.sleep(0.01)
    if Distence>300:
        return 0
    else:
        return Distence
def move(ch):
    if ch == 'w':
        Motor_Stop()
        time.sleep(revelocity)
        Motor_Forward()
        time.sleep(velocity)
        Motor_Stop()
        time.sleep(revelocity)
        Motor_Forward()
        time.sleep(velocity)
        Motor_Stop()
        time.sleep(revelocity)
        Motor_Forward()
        time.sleep(velocity)
        Motor_Stop()
        time.sleep(revelocity)
        Motor_Forward()
        time.sleep(velocity)

    elif ch == 'a':
        Motor_Stop()
        time.sleep(revelocity)
        Motor_TurnLeft()
        time.sleep(velocity)
        Motor_Stop()
        time.sleep(revelocity)
        Motor_TurnLeft()
        time.sleep(velocity)
        Motor_Stop()
        time.sleep(revelocity)
        Motor_TurnLeft()
        time.sleep(velocity)
        Motor_Stop()
        time.sleep(revelocity)
        Motor_TurnLeft()
        time.sleep(velocity)
    elif ch == 'd':
        Motor_Stop()
        time.sleep(revelocity)
        Motor_TurnRight()
        time.sleep(velocity)
        Motor_Stop()
        time.sleep(revelocity)
        Motor_TurnRight()
        time.sleep(velocity)
        Motor_Stop()
        time.sleep(revelocity)
        Motor_TurnRight()
        time.sleep(velocity)
        Motor_Stop()
        time.sleep(revelocity)
        Motor_TurnRight()
        time.sleep(velocity)
# def sendCarState():
#     while True:
#         Distance_1 = Get_Distance(1)
#         Distance_2 = Get_Distance(2)

def getSensorSignal():
    if GPIO.input(IR_L) == False:
        leftSensor = 0
    else :
        leftSensor = 1
    if GPIO.input(IR_R) == False:
        rightSensor = 0
    else:
        rightSensor = 1
    if GPIO.input(IR_M) == False:
        midSensor = 0
    else:
        midSensor = 1
    return [leftSensor, rightSensor, midSensor]
def recordCarState():
    global TIME
    global t2
    while True:
        if ACT != 'c':
            #print('%s acquire lock...' % threading.currentThread().getName())
            if lock.acquire():
                #print('%s get the lock.' % threading.currentThread().getName())
                sensorResult = getSensorSignal()
                Distance_1 = Get_Distance(1)
                Distance_2 = Get_Distance(2)
                TIME = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                recordfile = open('SensorDataRecord.txt', 'a')
                recordfile.write(TIME + '\t' + str(sensorResult[0]) + '\t\t' + str(sensorResult[1]) + '\t\t' + str(sensorResult[2]) + '\t\t' \
                    + str(int(Distance_1)) + '\t\t' + str(int(Distance_2)) + '\t\t' + ACT + '\n')
                recordfile.close()
                #print('%s release lock...' % threading.currentThread().getName())
                lock.release()
                time.sleep(0.1)
        else:
            print('record end')
            stop_thread(t2)
            break
def reciveCommand():
    global ACT
    global t1
    while True:
        if ACT != 'c' :
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)

            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            #print('%s acquire lock...' % threading.currentThread().getName())
            if lock.acquire():
                #print('%s get the lock.' % threading.currentThread().getName())
                if ch == 'w':
                    print(TIME+ ':' +'forward')
                    Motor_Forward()
                elif ch == 's':
                    print(TIME+ ':' +'stop')
                    Motor_Stop()
                elif ch == 'a':
                    print(TIME+ ':' +'left')
                    Motor_TurnLeft()
                elif ch == 'd':
                    print(TIME+ ':' +'right')
                    Motor_TurnRight()
                ACT=ch
                #print('%s release lock...' % threading.currentThread().getName())
                lock.release()
            else :
                print('cutdown!!!')
                stop_thread(t1)
                break


def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)
if __name__ == '__main__':
    TIME = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    recordfile = open('SensorDataRecord.txt', 'w')
    recordfile.write(TIME + '\t' + 'S_L' + '\t\t' + 'S_M' + '\t\t' + 'S_R' + '\t\t' + 'Dis_1' + '\t\t' + 'Dis_2' + '\t\t' +'Act' + '\n')
    recordfile.close()
    print('car ready!!')
    lock = threading.Lock()
    ACT = 'w'
    threads = []
    t1 = threading.Thread(target=reciveCommand)
    threads.append(t1)
    t2 = threading.Thread(target=recordCarState)
    threads.append(t2)
    for i in threads:
        #i.setDaemon(True)
        i.start()
    for i in threads:
        i.join()
    print('all end')
    exit()





