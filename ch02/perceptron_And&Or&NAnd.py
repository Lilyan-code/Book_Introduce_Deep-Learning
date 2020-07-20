# 利用感知机实现与、或、与非门电路
def AND(x1, x2): # x1, x2为感知机的两个input
    w1, w2, theta = 0.5, 0.5, 0.7  # w1， w2为权重， theta为阈值
    res = x1 * w1 + x2 * w2 # 当阈值theta < res(权重和）， 那么激活感知，反之不激活
    if res <= theta:
        return 0
    else:
        return 1

def OR(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.3
    res = x1 * w1 + x2 * w2
    if res <= theta:
        return 0
    else:
        return 1

def NAND(x1, x2):
    w1, w2, theta = -0.5, -0.5, -0.7
    res = x1 * w1 + x2 * w2
    if res <= theta:
        return 0
    else:
        return 1

print('Test AND:')
print(AND(0, 0))
print(AND(0, 1))
print(AND(1, 0))
print(AND(1, 1))

print('Test OR:')
print(OR(0, 0))
print(OR(0, 1))
print(OR(1, 0))
print(OR(1, 1))

print('Test NAND:')
print(NAND(0, 0))
print(NAND(0, 1))
print(NAND(1, 0))
print(NAND(1, 1))