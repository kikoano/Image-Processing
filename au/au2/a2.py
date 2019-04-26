import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.cvtColor(cv2.imread("moon.jpg"),cv2.COLOR_BGR2GRAY)

eqImg = cv2.equalizeHist(img)

cv2.imshow("Original image",img)
cv2.waitKey(0)
cv2.imshow("After histogram equalization",eqImg)
cv2.waitKey(0)
cv2.imwrite("Histogram_Equalization_eqImg.png",eqImg)

plt.figure(1)
plt.subplot(211)
hist, bins = np.histogram(img.flatten(),256,[0,256])
plt.hist(img.flatten(),256,[0,256],color="r")
plt.xlim([0,256])
plt.legend(("histogram"),loc="upper left")

plt.subplot(212)
hist, bins = np.histogram(eqImg.flatten(),256,[0,256])
plt.hist(eqImg.flatten(),256,[0,256],color="r")
plt.xlim([0,256])
plt.legend(("histogram"),loc="upper left")
plt.show()

cv2.destroyAllWindows()