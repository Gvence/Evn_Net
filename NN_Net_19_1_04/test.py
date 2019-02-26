import numpy as np

Data = np.array([100,8,9,7,6,4,2,8,3]).reshape((9,1))
print(Data)
m, n = np.shape(Data)
Result = []
for i in range(n):
    print(Data[:, i].T)
    max = Data[:, i].max()
    min = Data[:, i].min()
    avg = np.mean(Data[:, i], axis=0)
    print(avg)
    print((Data[:, i] - avg)/(max -min))
    Result.append((Data[:, i] - avg)/(max -min))
print(np.array(Result).T)