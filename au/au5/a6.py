# Example: chain link fence hole detection
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load test image and binarize
img = cv2.imread('fence.jpg', 0)
ret,BW = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Perform erosion with cross structuring element
se = cv2.getStructuringElement(cv2.MORPH_CROSS,(151,151))
BW1 = cv2.erode(BW,se,iterations = 1)

# Show images
plt.subplot(1, 4, 1)
plt.imshow(se,cmap='gray')
plt.title('Cross structuring element')

plt.subplot(1, 4, 2)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 4, 3)
plt.imshow(BW, cmap='gray')
plt.title('Binarized Image')

plt.subplot(1, 4, 4)
plt.imshow(BW1, cmap='gray')
plt.title('Eroded Image')

plt.show()

# Save images
cv2.imwrite('Binary_Erosion_Fence_bw-1.png', BW)
cv2.imwrite('Binary_Erosion_Fence_detect-1.png', BW1)

