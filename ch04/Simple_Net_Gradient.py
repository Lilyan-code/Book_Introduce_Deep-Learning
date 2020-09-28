import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # 将权重weight初始化为高斯分布2*3的数组

    # 前向传播
    def predict(self, x):
        return np.dot(x, self.W)
    # compute loss
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss



if __name__ == "__main__":
    net = simpleNet()
    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print('net output: ', p)
    t = np.array([0, 0, 1]) # label
    print('loss: ', net.loss(x, t))
    f = lambda w: net.loss(x, t)
    dW = numerical_gradient(f, net.W)
    print('Gradient is ', dW)