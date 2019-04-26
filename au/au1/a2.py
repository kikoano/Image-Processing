import cv2
from matplotlib import pyplot as plt

img = cv2.imread("messi5.jpg",cv2.IMREAD_GRAYSCALE)
plt.imshow(img,cmap="gray",interpolation="bicubic")
plt.xticks([])
plt.yticks([])
plt.show()