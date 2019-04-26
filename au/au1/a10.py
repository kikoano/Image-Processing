import cv2
import numpy as np
from matplotlib import pyplot as plt

maskImg = cv2.imread("mask.jpg",0)
liveImg = cv2.imread("live.jpg",0)

height, width = maskImg.shape
diffImg = np.zeros((height,width),np.uint8)
cv2.absdiff(maskImg,liveImg,diffImg)

clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
histeqDiffImg = clahe.apply(diffImg)

plt.figure(1)
plt.subplot(141)
plt.imshow(liveImg,cmap="gray")
plt.title("Live image")

plt.subplot(142)
plt.imshow(maskImg,cmap="gray")
plt.title("Mask image")

plt.subplot(143)
plt.imshow(diffImg,cmap="gray")
plt.title("Diffrence image")

plt.subplot(144)
plt.imshow(histeqDiffImg,cmap="gray")
plt.title("Histogram equalized difference image")
plt.show()

cv2.imwrite("test1.jpg",diffImg)
cv2.imwrite("test.jpg",histeqDiffImg)