# 朴素的梯度下降
import numpy as np
import matplotlib.pyplot as plt

def numerical_gradient_no_batch(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  # 生成一个与x的形状相同的且元素全为0的数组

    for idx in range(x.size):
        tmp_val = x[idx]  # 临时变量保存马上要update的x[idx]
        # compute f（x + h）
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # compute f(x - h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val # 将数组还原没有经过改变的值，方便下一次的迭代更新

    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):  # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
            grad[idx] = numerical_gradient_no_batch(f, x)

        return grad

# f: optimize function init_x:初始值  lr:learning rate step_num:梯度下降执行次数
def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x

def function2(x):
    return x[0] ** 2 + x[1] ** 2

init_x = np.array([-3.0, 4.0])
print('gradient descent: ', gradient_descent(function2, init_x = init_x, lr = 0.1, step_num = 100))