import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("moon.jpg",0)

eqImg = cv2.equalizeHist(img)

clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
claheImg1 = clahe.apply(img)

clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(16,16))
claheImg2 = clahe.apply(img)

plt.figure(1)

plt.subplot(221)
plt.imshow(img,cmap="gray")
plt.title("Original iamge")

plt.subplot(222)
plt.imshow(eqImg,cmap="gray")
plt.title("Global histogram")

plt.subplot(223)
plt.imshow(claheImg1,cmap="gray")
plt.title("Tiling 8*8 histograms")

plt.subplot(224)
plt.imshow(claheImg2,cmap="gray")
plt.title("Tiling 16*16 histograms")

plt.show()

cv2.imwrite('CLAHE_global.png',eqImg)
cv2.imwrite('CLAHE_8by8.png', claheImg1)
cv2.imwrite('CLAHE_16by16.png', claheImg2)