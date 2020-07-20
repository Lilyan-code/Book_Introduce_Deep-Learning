# Netrual Network Forward Propagation 神经网络前向传播算法
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['B1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['B2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['B3'] = np.array([0.1, 0.2])
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    B1, B2, B3 = network['B1'], network['B2'], network['B3']
    # comput one layer
    A1 = np.dot(x, W1) + B1
    Z1 = sigmoid(A1)

    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid(A2)

    A3 = np.dot(Z2, W3) + B3
    output = A3
    return output

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)