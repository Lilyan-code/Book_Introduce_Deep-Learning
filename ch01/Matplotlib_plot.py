import numpy as np
import matplotlib.pyplot as plt

# 生成数据
x = np.arange(0, 6, 0.1) # 生成[0, 0.1, 0.2, .... 5.9]
y1 = np.sin(x)
y2 = np.cos(x)

# 绘图
print('绘制sin函数 and cos函数')
plt.plot(x, y1, label = "sin")
plt.plot(x, y2, linestyle = "--", label = "cos") # 用虚线画出cos
plt.xlabel("x")
plt.ylabel("y")
plt.title("sin & cos function plot")
plt.legend() # 显示每一条line代表的种类
plt.show()

