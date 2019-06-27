import cv2
import mahotas
import numpy as np
import pickle
import csv
import glob
import os
from segment import segment

srcPath = ["database/", "query/"]
index = {}

def zernike_moments(img):
    # Segment
    seg = segment(img)
    outline = np.zeros(img.shape, dtype="uint8")
    # Find contours of segment image, sort them and draw them in outline
    cnts, _ = cv2.findContours(seg.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    cv2.drawContours(outline, [cnts], -1, 255, -1)
    # Do zernike moments on outline
    return mahotas.features.zernike_moments(outline, radius=max(img.shape)/2, degree=15)

for path in srcPath:
    for imagePath in glob.glob(path + "*jpg"):
        img = cv2.imread(imagePath, 0)
        res = zernike_moments(img)
        index[imagePath] = res

# Save to text
writer = csv.writer(open("zernikeMoments.txt", 'w'))
for key, val in index.items():
    writer.writerow([key, val])

# Saves object so it can be used in dom4
with open('index.pickle', 'wb') as file:
    pickle.dump(index, file)