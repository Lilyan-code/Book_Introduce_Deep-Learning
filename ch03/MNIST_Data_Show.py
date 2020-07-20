import sys, os
import  numpy as np
sys.path.append(os.pardir) # 导入父目录的文件
from dataset.mnist import load_mnist # 加载MINIST数据集
from PIL import Image # PIL（Python Image Library）

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

'''
Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
    
flattern:设置是否将输入展开（将图像展开为一维数组， 不展开为1x28x28三维图像） True：展开， False不展开
normalize:是否将图像正规化为0.0~1.0的值。False：图像按原来的像素保持0~255， True：正规化 
'''

(x_trian, t_trian), (x_test, t_test) = load_mnist(flatten = True, normalize = False)

img = x_trian[0]
label = t_trian[0]
print(label)
print(img.shape)
img = img.reshape(28, 28)
print(img.shape)
img_show(img)

print(x_trian.shape)
print(t_trian.shape)
print(x_test.shape)
print(t_test.shape)
