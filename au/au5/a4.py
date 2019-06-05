# Example: blob separation/ detection by erosion
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load test image
img = cv2.imread('circles.png');

# Perform erosion with square
se1 = cv2.getStructuringElement(cv2.MORPH_RECT,(30,30))
BW1 = cv2.erode(img,se1,iterations = 1)

# Perform erosion with square
se2 = cv2.getStructuringElement(cv2.MORPH_RECT,(70,70))
BW2 = cv2.erode(img,se2,iterations = 1)

# Perform erosion with square
se3 = cv2.getStructuringElement(cv2.MORPH_RECT,(96,96))
BW3 = cv2.erode(img,se3,iterations = 1)

# Show images
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(BW1)
plt.title('Eroded by Square of Width 30')

plt.subplot(2, 2, 3)
plt.imshow(BW2)
plt.title('Eroded by Square of Width 70')

plt.subplot(2, 2, 4)
plt.imshow(BW3)
plt.title('Eroded by Square of Width 96')

plt.show()
# Save images
cv2.imwrite('Binary_Erosion_Coins_square_30-1.png', BW1)
cv2.imwrite('Binary_Erosion_Coins_square_70-1.png', BW2)
cv2.imwrite('Binary_Erosion_Coins_square_90-1.png', BW3)

# Perform erosion with circle
se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
BW1 = cv2.erode(img,se1,iterations = 1)

# Perform erosion with circle
se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(35,35))
BW2 = cv2.erode(img,se2,iterations = 1)

# Perform erosion with circle
se3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(45,45))
BW3 = cv2.erode(img,se3,iterations = 2)

# Show images
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(BW1)
plt.title('Eroded by Disk of Radius 15')

plt.subplot(2, 2, 3)
plt.imshow(BW2)
plt.title('Eroded by Disk of Radius 35')

plt.subplot(2, 2, 4)
plt.imshow(BW3)
plt.title('Eroded by Disk of Radius 45')

plt.show()
# Save images
cv2.imwrite('Binary_Erosion_Coins_disk_15-1.png',BW1)
cv2.imwrite('Binary_Erosion_Coins_disk_35-1.png',BW2)
cv2.imwrite('Binary_Erosion_Coins_disk_45-1.png',BW3)