import numpy as np
import matplotlib.pyplot as plt

# RELU函数当input>0: output: x, input < 0, output = 0
def relu(x):
    return np.maximum(0, x) # maximum从输入的数值中选择较大的那个值进行输出

def step_function(x):
    return np.array(x > 0, dtype = np.int)

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid_function(x)
y2 = step_function(x)
y3 = relu(x)
plt.plot(x, y1, label = "sigmoid")
plt.plot(x, y2, label = "step", linestyle = "--")
plt.plot(x, y3, label = "relu", linestyle = "-")
plt.ylim(-0.1, 5.1)
plt.legend()
plt.show()
