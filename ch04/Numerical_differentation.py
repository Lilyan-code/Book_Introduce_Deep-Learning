# 数值微分求导数， 数值微分就是用数值方法近似的求解函数的导数的过程
import numpy as np
import matplotlib.pyplot as plt

# f = function, x为input

def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

# example 1: f = 0.01x^2 + 0.1x
def function1(x):
    return 0.01 * (x ** 2) + 0.1 * x

# 此函数用于利用数值微分的值作为斜率，重新画出与对应函数的斜线
def tangent_line(f, x):
    d = numerical_diff(f, x) # 计算导数
    print(d)
    y = f(x) - d*x # f(x)与直线的截距
    return lambda t: d*t + y # 类似返回一个直线函数 y = k * x + b

def function(x):
    return 0.02 * x + 0.1

# define 二元函数f(x0, x1) = x0^2 + x1^2
def function2(x):
    return np.sum(x ** 2)

# 当x0 = 3， x1 = 4时， 求funciton2对x0的偏微分
def function2_x0(x0):
    return x0**2 + 4.0 ** 2.0
print('当x0 = 3， x1 = 4时， 求funciton2对x0的偏微分: ', numerical_diff(function2_x0, 3.0))

# 当x0 = 3， x1 = 4时， 求funciton2对x1的偏微分
def function2_x1(x1):
    return 3.0 ** 2.0 + x1 ** 2
print('当x0 = 3， x1 = 4时， 求funciton2对x1的偏微分: ', numerical_diff(function2_x1, 4.0))


x = np.arange(0.0, 20.0, 0.1)
y = function1(x)
tf = tangent_line(function1, 5)
y1 = tf(x)
y2 = function(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.plot(x, y1)
plt.plot(x, y2)
plt.show()

# 如果通过我们自己计算导数f(x)的导数为0.02*x + 0.1
# 当x in 5： 0.02 * 5 + 0.1 = 0.2
# 当x in 10：0.02 * 10 + 0.1 = 0.3
# compare to this result, discover similarly
print('x in 5的导数：', numerical_diff(function1, 5))

print('x in 10的导数: ', numerical_diff(function1, 10))