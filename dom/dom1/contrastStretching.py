import cv2
import numpy as np
import numba
import time
from matplotlib import pyplot as plt

@numba.jit(nopython=True,cache=True)
def contrastStretching(img, x1, x2, y1, y2):
    ics = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            x = img[i, j]
            if x >= 0 and x < x1:
                ics[i, j] = y1/x1*x
            elif x >= x1 and x < x2:
                ics[i, j] = (y2-y1)/(x2-x1)*(x-x1)+y1
            elif x >= x2 and x <= 255:
                ics[i, j] = (255-y2)/(255-x2)*(x-x2)+y2

    return ics


imagePath = input("Enter image path: ")

x1 = int(input("Enter x1: "))
x2 = int(input("Enter x2: "))
y1 = int(input("Enter y1: "))
y2 = int(input("Enter y2: "))

img = cv2.imread(imagePath, 0)

# Old slow code
'''
start_time = time.time()
h, w = img.shape
ics = np.zeros((h, w), np.uint8)
for i in range(1, h):
    for j in range(1, w):
        x = img.item(i, j)
        if x >= 0 and x < x1:
            ics.itemset((i, j), y1/x1*x)
        elif x >= x1 and x < x2:
            ics.itemset((i, j), (y2-y1)/(x2-x1)*(x-x1)+y1)
        elif x >= x2 and x <= 255:
            ics.itemset((i, j), (255-y2)/(255-x2)*(x-x2)+y2)
elapsed_time = time.time() - start_time
print(elapsed_time)
'''
# Numpy code
'''
start_time = time.time()
ics = np.zeros_like(img)
low_band = (img >=0) & (img < x1)
ics[low_band] = y1/x2 * img[low_band]
middle_band = (img >= x1) & (img < x2)
ics[middle_band] = (y2-y1)/(x2-x1)*(img[middle_band]-x1)+y1
high_band = (img >= x2) & (img <= 255)
ics[high_band] = (255-y2)/(255-x2)*(img[high_band]-x2)+y2
elapsed_time = time.time() - start_time
print(elapsed_time)
'''

start_time = time.time()
# Numba fastest code compiles to c code with cache
ics=contrastStretching(img,x1,x2,y1,y2)
elapsed_time = time.time() - start_time
print("Contrast Stretching compile time: "+ str(elapsed_time))

cv2.imwrite("contrast.png", ics)

fig = plt.gcf()
fig.canvas.set_window_title("Contrast Stretching")
plt.figure(1)

plt.subplot(221)
plt.xticks([])
plt.yticks([])
plt.title("Original")
plt.imshow(img, "gray")

plt.subplot(222)

plt.hist(img.ravel(), 256, [0, 256])

plt.subplot(223)
plt.xticks([])
plt.yticks([])
plt.title("Contrast Stretching")
plt.imshow(ics, "gray")

plt.subplot(224)
plt.hist(ics.ravel(), 256, [0, 256])

plt.show()