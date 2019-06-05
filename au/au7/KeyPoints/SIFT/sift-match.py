import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('../images/example1.jpg',0)
img2 = cv2.imread('../images/example2.jpg',0)

sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
(kps1, descs1) = sift.detectAndCompute(img1, None)
(kps2, descs2) = sift.detectAndCompute(img2, None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(descs1,descs2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kps1,img2,kps2,good,None,flags=2)
plt.imshow(img3),plt.show()