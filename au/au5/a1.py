# Dilation
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load test image
img = cv2.imread('bacteria.png')

# Perform dilation with small disk
se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
BW1  = cv2.dilate(img,se1,iterations = 1)

# Perform dilation with larger disk
se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
BW2  = cv2.dilate(img,se2,iterations = 1)

# Show images
plt.subplot(131)
plt.imshow(img)
plt.title('Original Image')


plt.subplot(132)
plt.imshow(BW1)
plt.title('Dilation by Small Disk')

plt.subplot(133)
plt.imshow(BW2)
plt.title('Dilation by Larger Disk')

plt.show()

# Save images
cv2.imwrite('Dilation_disk_10-1.png', BW1)
cv2.imwrite('Dilation_disk_20-1.png', BW2)