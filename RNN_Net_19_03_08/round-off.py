import numpy as np
x = np.arange(0,10)/10 #+ 0.1
print('x:', x.reshape((2,5)))
print('result x :', np.around(x,0))