# pyplot还提供了imshow()显示图像， 使用matplotlib.image模块中的imread()方法读入图像
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('lena.png')
plt.imshow(img)

plt.show()