import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("bay.jpg")

plt.figure(1)
plt.subplot(1,2,1)
hist, bins = np.histogram(img.flatten(),256,[0,256])
plt.hist(img.flatten(),256,[0,256],color="r")
plt.xlim([0,256])
plt.legend(("histogram"),loc="upper left")
plt.title("Image histogram")

plt.subplot(1,2,2)
plt.title("Original image")
plt.imshow(img)

plt.show()
