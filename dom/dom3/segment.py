import cv2
import glob
import numpy as np
import os

srcPath = ["database/", "query/"]
destPath = "segmented/"

# Segment image
def segment(img):
    # Gaussian blur for better segment
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    # Threshold
    ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Close
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, se)
    # Open
    res = cv2.morphologyEx(closed, cv2.MORPH_OPEN, se)
    return res

# Save image
def saveImg(src_path, img):
    dest = destPath + src_path.replace("\\", "/")
    cv2.imwrite(dest, img)

if not os.path.exists(destPath):
    os.mkdir(destPath)
if not os.path.exists(destPath+srcPath[0]):
    os.mkdir(destPath+srcPath[0])
if not os.path.exists(destPath+srcPath[1]):
    os.mkdir(destPath+srcPath[1])

for path in srcPath:
    for imagePath in glob.glob(path + "*jpg"):
        # Gray image
        img = cv2.imread(imagePath, 0)
        seg = segment(img)
        saveImg(imagePath, seg)