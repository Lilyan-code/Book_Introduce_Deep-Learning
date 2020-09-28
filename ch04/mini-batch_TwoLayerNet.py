import numpy as np
from dataset.mnist import load_mnist
from Two_Layer_Net import TwoLayerNet
import matplotlib.pyplot as plt


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
iteration = []

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
lr = 0.1

net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    iteration.append(i)
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = net.numerical_gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        net.params[key] -= lr * grad[key]

    loss = net.loss(x_batch, t_batch)
    train_loss_list.append(loss)

plt.figure()
plt.plot(iteration, train_loss_list)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.legend()
plt.show()