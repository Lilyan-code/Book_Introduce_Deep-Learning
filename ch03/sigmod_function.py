import matplotlib.pyplot as plt
import numpy as np

def step_function(x):
    return np.array(x > 0, dtype = np.int)

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid_function(x)
y2 = step_function(x)
plt.plot(x, y1, label = "sigmoid")
plt.plot(x, y2, label = "step", linestyle = "--")
plt.ylim(-0.1, 1.1)
plt.legend()
plt.show()

