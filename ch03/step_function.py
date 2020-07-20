# 画出阶跃函数的图形，如图，成阶梯形变化，以0为界
import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return np.array(x > 0, dtype = np.int) # 返回一个NumPy中的数组，先判断x是否大于0，返回一个bool数组，然后再强转为只有0， 1的int数组

x = np.arange(-5.0, 5.0, 0.1) # generate array[-5.0, -4.9...., 4.9]
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # 指定y轴范围
plt.show()