import cv2
import numpy as np

x = np.uint8([250])
y = np.uint8([10])

print(cv2.add(x,y))

print (x+y)

img1 = cv2.imread("ml.jpg")
img2 = cv2.imread("opencv_logo.png")

dst = cv2.addWeighted(img1,0.7,img2,0.3,0)

cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()