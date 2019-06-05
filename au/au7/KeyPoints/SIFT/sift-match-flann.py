import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('../images/example1.png',0)
img2 = cv2.imread('../images/example2.png',0)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
(kps1, descs1) = sift.detectAndCompute(img1, None)
(kps2, descs2) = sift.detectAndCompute(img2, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(descs1,descs2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in xrange(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
	if m.distance < 0.7*n.distance:
		matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), matchesMask = matchesMask, flags = 0)
img3 = cv2.drawMatchesKnn(img1,kps1,img2,kps2,matches,None,**draw_params)
plt.imshow(img3,),plt.show()