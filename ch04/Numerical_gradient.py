# 梯度的实现
import numpy as np
import matplotlib.pyplot as plt
# f(x0, x1) = x0 ** 2 + x1 ** 2
def function2(x):
    if x.ndim == 1:
        return np.sum(x ** 2)
    else:
        return np.sum(x ** 2, axis=1)

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


def function_2(x):
    if x.ndim == 1:
        return np.sum(x ** 2)
    else:
        return np.sum(x ** 2, axis=1)


def tangent_line(f, x):
    d = numerical_gradient(f, x)
    print(d)
    y = f(x) - d * x
    return lambda t: d * t + y


if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)  # meshgrid用两个坐标轴上的点画图，默认返回笛卡尔积的形式

    X = X.flatten()  # 返回一个一维数组
    Y = Y.flatten()

    grad = numerical_gradient(function_2, np.array([X, Y])) # np.array([X, Y]): 返回一个由X， Y组成的二维数组

    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")  # ,headwidth=10,scale=40,color="#444444")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()

    print('求（3， 4）这个点的gradient：', numerical_gradient_no_batch(function2, np.array([3.0, 4.0])))

    print('求（0， 2）这个点的gradient：', numerical_gradient_no_batch(function2, np.array([0.0, 2.0])))

    print('求（3， 0）这个点的gradient：', numerical_gradient_no_batch(function2, np.array([3.0, 0.0])))