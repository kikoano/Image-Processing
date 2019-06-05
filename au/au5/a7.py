# Small hole removal by closing
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load test image
img = cv2.imread('peter.png', 0)

# Binarize image
ret,origMask = cv2.threshold(img,120,255,cv2.THRESH_BINARY)

# Perform closing
se = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))

img_dilated = cv2.dilate(origMask,se,iterations = 1)
img_closed = cv2.morphologyEx(origMask, cv2.MORPH_CLOSE, se)
#opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

img_dif = img_closed - origMask ;

height, width = img.shape
img_dif = np.zeros((height,width), np.uint8)
cv2.absdiff(img_closed,origMask,img_dif)

# Show images
plt.subplot(2, 2, 1)
plt.imshow(origMask, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(img_dilated, cmap='gray')
plt.title('Dilated Image')

plt.subplot(2, 2, 3)
plt.imshow(img_closed, cmap='gray')
plt.title('Closed Image')

plt.subplot(2, 2, 4)
plt.imshow(img_dif, cmap='gray')
plt.title('Original - Closed')

plt.show()
# Save images
cv2.imwrite('Small_Hole_Removal_original-1.png', origMask)
cv2.imwrite('Small_Hole_Removal_dilation-1.png', img_dilated)
cv2.imwrite('Small_Hole_Removal_closing-1.png', img_closed)
cv2.imwrite('Small_Hole_Removal_difference-1.png', img_dif)