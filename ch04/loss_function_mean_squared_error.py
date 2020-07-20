# loss function损失函数常用的mean squared error（均方误差） 0.5 * (for i in range(n): np.sum((yi - ti) ** 2)
import numpy as np
# y: softmax function output
# t：test data set label
def mean_squared_error(y , t):
    return 0.5 * np.sum((y - t) ** 2)

# 人为设定test label为2，即当前标签为digit 2
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# example 1
# y1中代表softmax predict 从0～9的概率
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print('y1: ', mean_squared_error(np.array(y1), np.array(t)))

# example 2
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print('y2: ', mean_squared_error(np.array(y2), np.array(t)))

'''
由mean_squared_error可以看出第一个神经网络输出的结果比第二个神经网络的误差较小，所以example1更加接近正解
'''