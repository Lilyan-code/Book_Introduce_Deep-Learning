# 单层感知机只能处理线性的逻辑门，XOR则需要非线性处理，即多层感知机
# 将x1， x2作为NAND门以及OR门的输入，定义s1 = NAND output, s2 = OR output
# 将s1, s2作为AND门的input， output即是XOR门的输出
'''
The Summary of Chapter2:
1. 感知机是具有input和output的algorithm。给定一个输入后，将输出一个既定的值
2. 感知机将weight和bias设定为parameter
3. 使用感知机可以表示AND gate和OR gate等逻辑电路
4. XOR gate无法通过单层感知机来表示
5. 使用2层的感知机可表示异或门
6. 单层感知机只能表示线性空间，而多层感知机能表示非线性空间
7. 多层感知机（在理论上）可用于表示计算机
'''

import numpy as np
def AND(x1, x2): # 使用numpy完成权重和的计算
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    res = np.sum(w * x) + b
    if res <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.3
    res = np.sum(w * x) + b
    if res <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    res = np.sum(w * x) + b
    if res <= 0:
        return 0
    else:
        return 1

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    output = AND(s1, s2)
    return output

print('XOR门 输出')
print(XOR(0, 0))
print(XOR(0, 1))
print(XOR(1, 0))
print(XOR(1, 1))