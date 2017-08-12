#encoding: utf8

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pdb

img = Image.open('lettera.bmp')
img = np.array(img)
img1 = img
if img.ndim == 3:
    img1 = img[:,:,0]

img2 = np.copy(img1)
img2[img2 < 128] =  0
img2[img2 >= 128] = 255

plt.subplot(221); plt.imshow(img1)
plt.subplot(222); plt.imshow(img1, cmap ='gray')
plt.subplot(223); plt.imshow(img1, cmap = plt.cm.gray)
plt.subplot(224); plt.imshow(img2, cmap = plt.cm.gray)
plt.savefig('images/lettera.png', format='png')
