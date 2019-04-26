import cv2
import numpy as np
from matplotlib import pyplot as plt

maskImg1 = cv2.imread("mask1.png",0)
maskImg2 = cv2.imread("mask2.png",0)

plt.figure(1)

plt.subplot(131)
plt.imshow(maskImg1,cmap="gray")
plt.subplot(132)
plt.imshow(maskImg2,cmap="gray")

height, width = maskImg1.shape
diffImg = np.zeros((height,width),np.uint8)
cv2.absdiff(maskImg1,maskImg2,diffImg)
plt.subplot(133)

plt.imshow(diffImg,cmap="gray")
plt.title("Difference Image")
plt.show()
cv2.imwrite("Mask_Comparison_diff.png",diffImg)