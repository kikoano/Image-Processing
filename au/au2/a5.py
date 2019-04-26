import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("test1.jpg")
print(img.shape)

mask = np.zeros(img.shape[:2],np.uint8)
mask[400:1500, 600:2000] = 255
masked_img = cv2.bitwise_and(img,img,mask=mask)

hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])

plt.figure(1)
plt.subplot(121)
plt.imshow(img)

plt.subplot(122)
plt.imshow(masked_img)

plt.figure(2)
plt.imshow(mask,'gray')

plt.figure(3)
plt.plot(hist_full)
plt.plot(hist_mask)
plt.xlim([0,256])

plt.show()