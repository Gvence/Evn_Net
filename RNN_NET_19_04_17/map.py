# Self Driving Car

# Importing the libraries
import numpy as np
np.set_printoptions(threshold=np.inf)
import random
import matplotlib.pyplot as plt
import time
import os

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

# Importing the Dqn object from our AI in ai.py
from RL_brain import DeepQNetwork
from mixed_train_processing import LSTMRNN
# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')



# Initializing the map



# Initializing the last distance


# Creating the car class

class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)#传感器位置的坐标列表
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-sensitivity:int(self.sensor1_x)+sensitivity, int(self.sensor1_y)-sensitivity:int(self.sensor1_y)+sensitivity]))/pow(2.*sensitivity,2)#以传感器为圆心的20*20像素格内沙子的比例
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-sensitivity:int(self.sensor2_x)+sensitivity, int(self.sensor2_y)-sensitivity:int(self.sensor2_y)+sensitivity]))/pow(2.*sensitivity,2)
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-sensitivity:int(self.sensor3_x)+sensitivity, int(self.sensor3_y)-sensitivity:int(self.sensor3_y)+sensitivity]))/pow(2.*sensitivity,2)
        if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:#传感器探测到地图边沿
            self.signal1 = 1.
        if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
            self.signal2 = 1.
        if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
            self.signal3 = 1.

class Ball1(Widget):#三个代表传感器的球
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass

# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def init(self):
        global sand
        global goal_x
        global goal_y
        global first_update
        global sensitivity
        global last_y
        global last_x
        global n_points
        global length
        global action2rotation
        global last_reward
        global scores
        global line_num
        global line_record
        global point_record
        global last_distance
        # Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
        last_x = 0
        last_y = 0
        n_points = 0
        length = 0
        line_num = 0
        # Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
        self.brain = DeepQNetwork()  #
        action2rotation = [0, 20, -20]  # 三个方向，前进0，左拐20，右拐-20
        last_reward = 0
        last_distance = 0
        scores = []
        sensitivity = 1.5
        sensitivity = int(10 * sensitivity)
        sand = np.zeros((longueur, largeur))
        goal_x = 20
        goal_y = largeur - 20
        first_update = False
        if RECORD_LINE:
            file = open('point_record.txt', 'w')
            file.close()
            file = open('line_record.txt', 'w')
            file.close()
        else:
            line_record = np.loadtxt('line_record.txt')
            point_record = np.loadtxt('point_record.txt')
            print('read points OK! shape of points record:', point_record.shape)
            print('read lines OK! shape of lines record:', line_record.shape)
    def serve_car(self):
            self.car.center = self.center
            self.car.velocity = Vector(1, 1)



    def update(self, dt):

        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur

        longueur = self.width
        largeur = self.height
        if first_update:
            self.init()

        #orientation = Vector(*self.car.velocity).angle((xx,yy))/180.#小车速度 乘上小车向目标方向的修正等一现在小车朝着目标方向开
        # 将小车的正负运动方向和小车的传感器的回馈作为输入
        observation = [self.car.signal1, self.car.signal2, self.car.signal3, self.car.x, self.car.y]
        action = self.brain.choose_action(observation)#通过学习小车状态和环境回报选择方向
        rotation = action2rotation[action]#通过网络跑出的动作转换成小车的运动方向，左转还是右转还是前进
        #小车的实际状态在move后便发生改变，但我们将改变后的状态作为下一个状态保存，用于以后训练
        self.car.move(rotation)
        observation_ = [self.car.signal1, self.car.signal2, self.car.signal3, self.car.x, self.car.y]
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        if sand[int(self.car.x),int(self.car.y)] > 0:
            #self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            # self.car.x = self.width - goal_x
            # self.car.y = self.height - goal_y
            # self.car.x -= 10
            # self.car.y -= 10
            last_reward = -1
        else: # otherwise
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            last_reward = -0.2
            if distance < last_distance:
                last_reward = 0.01

        if self.car.x < 10:
            self.car.x = 10
            last_reward = -1
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            last_reward = -1
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            last_reward = -1

        if distance < 100:#小车会向右下角开当与右下角的距离《100时向左上角开周而复始，遇到障碍躲避
            goal_x = self.width-goal_x
            goal_y = self.height-goal_y
            last_reward += 0.1
        last_distance = distance
        #保存状态等信息
        self.brain.store_transition(observation, action, last_reward, observation_, Evn = True)
# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y,line_num,index,i
        with self.canvas:
            #记录障碍信息
            if RECORD_LINE:
                Color(0.8,0.7,0)
                d = 10.
                line_num += 1
                touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
                print('line_num', line_num)
                # 测试时记录point的位置
                file = open('point_record.txt', 'a')
                file.write(str(round(touch.x,1)) + '\t' + str(round(touch.y,1)) + '\t'+ str(line_num) + '\n')
                file.close()
                last_x = int(touch.x)
                last_y = int(touch.y)
                n_points = 0
                length = 0
                sand[int(touch.x),int(touch.y)] = 1
            #重写障碍信息
            elif line_num < len(point_record):
                Color(0.8, 0.7, 0)
                d = 10.
                line_num += 1
                i =  np.argwhere(line_record[:, 2] == line_num)[0]
                print(line_num, i)
                index = np.argwhere(point_record[:,2] == line_num)[0]
                touch.ud['line'] = Line(points=(point_record[index, 0], point_record[index, 1]), width=10)
                print('line_num', line_num)
                last_x = int(point_record[index, 0])
                last_y = int(point_record[index, 1])
                n_points = 0
                length = 0
                sand[int(point_record[index, 0]), int(point_record[index, 1])] = 1
            else:
                print('no more line!!')

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y,i
        if touch.button == 'left' :
            # 记录障碍信息
            if RECORD_LINE:
                touch.ud['line'].points += [touch.x, touch.y]
                file = open('line_record.txt', 'a')
                file.write(str(round(touch.x,1)) + '\t' + str(round(touch.y,1)) + '\t'+ str(line_num) + '\n')
                file.close()
                x = int(touch.x)
                y = int(touch.y)
                length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
                n_points += 1.
                density = n_points/(length)
                touch.ud['line'].width = int(20 * density + 1)
                sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
                last_x = x
                last_y = y
            # 重写障碍信息
            elif line_num <= len(point_record):
                if i < np.argwhere(line_record[:,2] == line_num)[-1]:
                    i += 1
                    touch.ud['line'].points += [line_record[i,0], line_record[i,1]]
                    x = int(line_record[i,0])
                    y = int(line_record[i,1])
                    length += np.sqrt(max((x - last_x) ** 2 + (y - last_y) ** 2, 2))
                    n_points += 1.
                    density = n_points / (length)
                    touch.ud['line'].width = int(20 * density + 1)
                    sand[int(line_record[i,0]) - 10: int(line_record[i,0]) + 10, int(line_record[i,1]) - 10: int(line_record[i,1]) + 10] = 1
                    last_x = x
                    last_y = y


# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        global first_update
        global RECORD_LINE
        RECORD_LINE = False
        first_update = True
        self.parent = Game()
        self.Evn = LSTMRNN()
        self.parent.serve_car()
        Clock.schedule_interval(self.parent.update, 1.0/60.0)
        Clock.schedule_interval(self.train_EvnNet, 1.0/60.0)
        Clock.schedule_interval(self.train_DQN, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (self.parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * self.parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        self.parent.add_widget(self.painter)
        self.parent.add_widget(clearbtn)
        self.parent.add_widget(savebtn)
        self.parent.add_widget(loadbtn)
        return self.parent
    def train_EvnNet(self,obj):
        if self.parent.brain.FIND_MEMORY_STACK:
            #如果brain网络没有处在保存模型阶段，并且Evn网络处于初始数据阶段，重新抽取一个memory
            if not self.parent.brain.saving_stack and self.Evn.INITSTATE and self.parent.brain.read_OK:
                 dic = self.parent.brain.read_memorystack()
                 self.Evn.get_batch(dic[str(np.random.randint(0,len(dic)))])

            #如果Evn网络没有处于初始数据和训练数据阶段，训练Evn网络
            if not self.Evn.INITSTATE and not self.Evn.TRAINING:
                 self.Evn.train()
        else:
            pass

    def train_DQN(self,obj):
        if self.parent.brain.FIND_MEMORY_STACK:
            #第一次训练随机选取一个状态
            if self.parent.brain.INITSTATE and not self.parent.brain.saving_stack and self.parent.brain.read_OK:
                self.GOAL_X = 20
                self.GOAL_Y = self.parent.width - 20
                self.last_distance = 0
                dic = self.parent.brain.read_memorystack()
                index = np.random.choice(self.parent.brain.memory_size, 1)
                self.M_S_ = np.array(dic[str(np.random.randint(0, self.parent.brain.stack_size))])[index, :5].flatten()
                self.parent.brain.INITSTATE = False
            if not self.parent.brain.INITSTATE:
                self.M_S = self.M_S_
                action = np.array([self.parent.brain.choose_action(self.M_S)]).flatten()
                input = np.concatenate((self.M_S , action)).reshape(-1,1,self.Evn.input_size)
                D_pred, S_pred = self.Evn.sess.run([self.Evn.pred_D, self.Evn.pred_S], feed_dict= {self.Evn.xs: input})
                self.M_S_ = np.concatenate((np.round(S_pred[-1, :],0), D_pred[-1, :]))
                distance = np.sqrt((D_pred[-1,0] - self.GOAL_X) ** 2 + (D_pred[-1,1] - self.GOAL_Y) ** 2)

                if int(S_pred[-1,0]) == 1 or  int(S_pred[-1,1]) == 1 or int(S_pred[-1,2]) == 1:
                    reward = -1
                else:  # otherwise
                    reward = 0
                    if distance < self.last_distance:
                        reward = 0.5

                if D_pred[-1,0] < 10:
                    D_pred[-1, 0] = 10
                    reward = -1
                if D_pred[-1,0] > self.parent.width - 10:
                    D_pred[-1, 0] = self.parent.width - 10
                    reward = -1
                if D_pred[-1,1] < 10:
                    D_pred[-1, 1] = 10
                    reward = -1
                if D_pred[-1,1] > self.parent.height - 10:
                    D_pred[-1, 1] = self.parent.height - 10
                    reward = -1

                if distance < 100:  # 小车会向右下角开当与右下角的距离《100时向左上角开周而复始，遇到障碍躲避
                    self.GOAL_X = self.parent.width - self.GOAL_X
                    self.GOAL_Y = self.parent.height - self.GOAL_Y
                    reward += 1
                self.last_distance = distance
                # 保存状态等信息
                self.parent.brain.store_transition(self.M_S.flatten(), action.flatten(), reward, self.M_S_.flatten(), M_Evn=True)
                if len(self.parent.brain.M_Evn_memory) > self.parent.brain.batch_size*2:
                    self.parent.brain.learn()
        else:
            pass
    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))
        file = open('point_record.txt', 'w')
        file.close()
        file = open('line_record.txt', 'w')
        file.close()

    def save(self, obj):
        print("saving brain...")
        self.parent.brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        self.parent.brainbrain.load()

# Running the whole thing
if __name__ == '__main__':
    App = CarApp()
    App.run()
