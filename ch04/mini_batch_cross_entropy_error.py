# 使用mini batch处理单个或多个数据，并进行交叉熵误差
import numpy as np
# 如何从数据集中随机选取十个data？
'''
这里的x_trian， t_data是在第三个例子中的MNIST的train data
train_size = x_trian.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
y_batch = t_train[batch_mask]
'''

# use one-hot way:
def one_hot_cross_entropy_error(y , t):
    if y.ndim == 1:  # 处理单个数据需要重塑矩阵形状
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    delta = 1e-7
    batch_size = y.shape[0] # 获得batch的size
    return -np.sum(t * np.log(y + delta)) / batch_size

# don't use one-hot way:
'''
当label不再是0， 1表示的时候，而是7，2这样表示的时候
np.arange(batch_size):表示的是从0-batch_size - 1的数组
假设t = [2, 7, 0, 9, 4]
那么y[np.arange(batch_size), t] = [y[0, 2], y[1, 7], y[2, 0], y[3, 9], y[4, 4]]
'''
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    delta = 1e-7
    batch_size = y.shape[0]

