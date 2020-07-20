# 输出神经元的激活函数，分类问题常用softmax， softmax = exp(Ak) / for i in range(1, n + 1) sum(Ai)
# softmax的分子是输入信号Ak的指数函数，分母是所有输入信号的指数函数和
import numpy as np
'''
softmax funciton:
softmax的输出范围都在0.0~1.0之间的实数
softmax的输出总和为1， 因为有这个缘故，我们可以把softmax的输出解释为概率
'''
# way 1: normalize:
def softmax_normal(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

# way 2: Multiply Contant C"
def softmax_optimize(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # 解决溢出问题
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y