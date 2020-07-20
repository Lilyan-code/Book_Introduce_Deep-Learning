# 将阈值转换为偏置实现AND、OR、NAND
# 在阈值的式子为 w1 * x1 + w2 * x2 <= theta 为0， w1 * x1 + w2 * x2 > theta 为1
# 令theta = -b, 通过移项可得到如下式子
# w1 * x1 + w2 * x2 + b <= 0 为 0， w1 * x1 + w2 * x2 + b > 0 为 1
# w1, w2是控制输入信号的重要性参数， 而bias偏置的值决定了激活神经元的难易程度
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

print('Test AND:')
print(AND(0, 0))
print(AND(0, 1))
print(AND(1, 0))
print(AND(1, 1))

print('Test OR:')
print(OR(0, 0))
print(OR(0, 1))
print(OR(1, 0))
print(OR(1, 1))

print('Test NAND:')
print(NAND(0, 0))
print(NAND(0, 1))
print(NAND(1, 0))
print(NAND(1, 1))