# Graylevel morphological edge detectors
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load test image
img = cv2.imread('bike.png', 0)

# Perform dilation
se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
img_dilated = cv2.dilate(img,se1,iterations = 1)

# Perform subtraction and thresholding
height, width = img.shape
edge = np.zeros((height,width), np.uint8)
cv2.absdiff(img_dilated,img,edge)
ret,BW = cv2.threshold(edge,20,255,cv2.THRESH_BINARY)

# Show images
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(img_dilated, cmap='gray')
plt.title('Dilated Image')

plt.subplot(2, 2, 3)
plt.imshow(edge, cmap='gray')
plt.title('Edge = Dilated - Original')

plt.subplot(2, 2, 4)
plt.imshow(BW, cmap='gray')
plt.title('Edge > Threshold')

plt.show()
# Save images
cv2.imwrite('Graylevel_Edge_Detection_dilated-1.png', img_dilated)
cv2.imwrite('Graylevel_Edge_Detection_edge-1.png', edge)
cv2.imwrite('Graylevel_Edge_Detection_bw_edge-1.png', BW)