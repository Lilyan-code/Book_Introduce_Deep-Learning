import numpy as np
from dataset.mnist import load_mnist
from Two_Layer_Net import TwoLayerNet
import matplotlib.pyplot as plt


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
iteration = []
train_acc_list = []
test_acc_list = []
epoch_list = []

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
lr = 0.1
iter_epoch = max(train_size / batch_size, 1)


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

    if i % iter_epoch == 0:
        train_acc =net.accuracy(x_train, t_train)
        test_acc = net.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('train acc, test acc | ' + str(train_acc) + ', ' + str(test_acc))

plt.figure()
plt.plot(epoch_list, train_acc_list, label='train')
plt.plot(epoch_list, test_acc_list, linestyle = '--', label='test')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()