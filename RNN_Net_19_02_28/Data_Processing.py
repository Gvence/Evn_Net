import os
import numpy as np

class dataRebuild(object):
    def __init__(self, dataFileName,trainRatio = 0.7, testRatio = 0.15):
        self.DFName = dataFileName
        self.TraR = trainRatio
        self.TesR = testRatio
        self.ValR = 1 - (trainRatio + testRatio)
        self.loadData()
        self.creatTrainData()
        self.creatTestData()
        self.creatValidationData()

    #加载数据
    def loadData(self):
        self.Data = np.loadtxt(self.DFName)
        print(self.Data[0, :])
        self.dataNum_H = self.Data.shape[0]
        self.dataNum_D = self.Data.shape[1]

    #创建训练集
    def creatTrainData(self):
        if self.TraR >0 :
            self.trainData_H = int(self.dataNum_H * self.TraR)
            file = open('trainData.txt', 'w')
            for i in self.Data[0 : self.trainData_H, :]:
                for j in i :
                    file.write('%.2f'%j + '\t')
                file.write('\n')
            file.close()
            print('Train data OK! SHAPE :[%s, %s]\n'%(self.trainData_H, self.dataNum_D))
        else:
            print('No train data! \n')

    # #创建测试集
    def creatTestData(self):
        if self.TesR > 0:
            self.testData_H = int(self.dataNum_H * self.TesR)
            file = open('testData.txt', 'w')
            for i in self.Data[self.trainData_H : self.trainData_H + self.testData_H, :]:
                for j in i :
                    file.write('%.2f'%j + '\t')
                file.write('\n')
            file.close()
            print('Test data OK! SHAPE :[%s, %s]\n'%(self.testData_H, self.dataNum_D))
        else:
            print('No test data! \n')

    #创建验证集
    def creatValidationData(self):
        if self.ValR > 0:
            self.valData_H = int(self.dataNum_H * self.ValR)
            file = open('valData.txt', 'w')
            for i in self.Data[self.trainData_H + self.testData_H : self.trainData_H + self.testData_H + self.valData_H, :]:
                for j in i :
                    file.write('%.2f'%j + '\t')
                file.write('\n')
            file.close()
            print('Validation data OK! SHAPE :[%s, %s]\n'%(self.valData_H, self.dataNum_D))
        else:
            print('No validation data! \n')

