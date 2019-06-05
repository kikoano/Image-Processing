# Morphological edge detectors
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load test image
img = cv2.imread('cliparts.png', 0)

# Extract edges
se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))

img_dilated = cv2.dilate(img,se1,iterations = 1)
img_eroded = cv2.erode(img,se1,iterations = 1)

height, width = img.shape
edge1 = np.zeros((height,width), np.uint8)
cv2.absdiff(img_dilated,img,edge1)

edge2 = np.zeros((height,width), np.uint8)
cv2.absdiff(img,img_eroded,edge2)

edge3 = cv2.add(edge1,edge2)

# Show images
plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title('Original')

plt.subplot(222)
plt.imshow(edge1, cmap='gray')
plt.title('Edge_1 = Dilated - Original')

plt.subplot(223)
plt.imshow(edge2, cmap='gray')
plt.title('Edge_2 = Original - Eroded')

plt.subplot(224)
plt.imshow(edge3, cmap='gray')
plt.title('Edge_1 + Edge_2')

plt.show()

cv2.imwrite('Edge_Detection_dilatedEdge-1.png', edge1)
cv2.imwrite('Edge_Detection_erodedEdge-1.png', edge2)
cv2.imwrite('Edge_Detection_combinedEdge-1.png', edge3)





