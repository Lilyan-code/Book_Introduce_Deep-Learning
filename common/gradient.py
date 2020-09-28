# coding: utf-8
import numpy as np

def _numerical_gradient_1d(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        
    return grad

# 二维偏导梯度
def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)
        
        return grad


def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    '''
    np.nditer:
    默认情况下，nditer将输入操作数视为只读对象。 为了能够修改数组元素，必须使用 'readwrite' 或 'writeonly' 每操作数标志指定读写或只写模式。
    Python迭代器协议没有从迭代器查询这些附加值的自然方法， 因此我们引入了一个替代语法来迭代nditer。 
    此语法显式使用迭代器对象本身，因此在迭代期间可以轻松访问其属性。使用此循环结构，可以通过索引到迭代器来访问当前值，
    并且正在跟踪的索引是属性 索引 或 multi_index， 具体取决于请求的内容。
    遗憾的是，Python交互式解释器在循环的每次迭代期间打印出while循环内的表达式的值。我们使用这个循环结构修改了示例中的输出，以便更具可读性
    '''
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        it.iternext()   
        
    return grad