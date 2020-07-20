# loss function常用的一种交叉熵函数cross_entropy_error, E = -(for k in range(len(y) np.sum(np.log(y) * t))
import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7  #定一个非常小的theta主要用于防止np.log(0)变为-inf，导致后面的数据无法正常计算
    return -(np.sum(t * np.log(y + delta)))

# 人为设定test label为2，即当前标签为digit 2
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# example 1
# y1中代表softmax predict 从0～9的概率
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print('y1: ', cross_entropy_error(np.array(y1), np.array(t)))

# example 2
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print('y2: ', cross_entropy_error(np.array(y2), np.array(t)))

'''
cross entropy error
由于t中的数据除了正确标签为1， 其余标签都为0，所以cross entropy error计算的是对应的正确标签的自然对数
由于对数函数log的性质，x = 1时， 由= 0， 随着x靠近0，y逐渐变小，由于cross entropy error带上负号，所以越接近0，误差越大
'''