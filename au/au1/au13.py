import cv2
import numpy as np
from matplotlib import pyplot as plt

origImg = cv2.imread("pcbCropped.png",0)

defectImg = cv2.imread("pcbCroppedTranslatedDefected.png",0)

height, width = origImg.shape

xShift =10
yShift = 10
registImg = np.zeros((height,width),np.uint8)
registImg[yShift : height, xShift : width] = defectImg[0 : height - yShift, 0 : width - xShift]

plt.figure(1)
diffImg1 = np.zeros((height,width),np.uint8)
cv2.absdiff(origImg,defectImg,diffImg1)

plt.subplot(131)
plt.imshow(diffImg1,cmap="gray")
plt.title("Unaligned Difference Image")

diffImg2 = np.zeros((height,width), np.uint8)
cv2.absdiff(origImg,registImg,diffImg2)

plt.subplot(132)
plt.imshow(diffImg2,cmap="gray")
plt.title("Aligned Difference Image")

ret, bwImg = cv2.threshold(diffImg2,20,255,cv2.THRESH_BINARY)

plt.subplot(133)
plt.imshow(bwImg,cmap="gray")
plt.title("Thresholded + Aligned Difference Image")
plt.show()

cv2.imwrite('Defect_Detection_diff.png', diffImg1)
cv2.imwrite('Defect_Detection_diffRegisted.png', diffImg2)
cv2.imwrite('Defect_Detection_bw.png', bwImg)