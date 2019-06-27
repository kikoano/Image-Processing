import numpy as np
import cv2
import pickle
import glob
import sys
from matplotlib import pyplot as plt

def pickleKeypoints(keypoints, descriptors):
    i = 0
    arr = []
    for point in keypoints:
        arr.append((point.pt, point.size, point.angle, point.response,
                point.octave, point.class_id, descriptors[i]))
        i += 1
    return arr

def unpickleKeypoints(arr):
    keypoints = []
    descriptors = []
    for point in arr:
        tempFeature = cv2.KeyPoint(x=point[0][0], y=point[0][1],_size=point[1], _angle=point[2],
                                    _response=point[3], _octave=point[4],_class_id=point[5])
        tempDescriptor = point[6]
        keypoints.append(tempFeature)
        descriptors.append(tempDescriptor)
    return keypoints, np.array(descriptors)

def resizeImg(original, newSize):
    coef = min(1, newSize / original.shape[1])
    return cv2.resize(original, None, fx=coef, fy=coef,interpolation=cv2.INTER_AREA)

def flann_knn_matcher(desc1, desc2):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)
    
    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append([m])
    return good

def search(path, index):
    # Load image 
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # Load gray image
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Resize both images
    img = resizeImg(img, 500)
    imgGray = resizeImg(imgGray, 500)

    # Keypoints and descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    (kps, descs) = sift.detectAndCompute(imgGray, None)

    # Minimum matching points
    MIN_MATCH_COUNT = 10

    result = {}
    max_inliers = 0

    for (k, v) in index.items():
        train_kps, train_desc = v
        good = flann_knn_matcher(descs, train_desc)
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kps[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([train_kps[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)

            _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            inliers_num = len([i for i in matchesMask if i == 1])
            
            if inliers_num > max_inliers:
                max_inliers = inliers_num
                result['img_path'] = k
                result['key_points'] = train_kps
                result['good_points'] = good
                result['mask'] = matchesMask

    for i in range(len(result['mask'])):
        if result['mask'][i] == 1:
            result['mask'][i] = [1, 0]
        else:
            result['mask'][i] = [0, 0]

    # Show original images
    result_img = cv2.imread( result['img_path'], cv2.IMREAD_COLOR)
    result_img = resizeImg(result_img, 300)
    img3 = cv2.drawMatches(img, None, result_img, None, None, None)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    plt.imshow(img3), plt.xticks([]), plt.yticks([])
    plt.show()

    # Show images with SIFT descriptors
    query_img_sift = cv2.drawKeypoints(img, kps, None, color=(0, 255, 255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    result_img_sift = cv2.drawKeypoints(result_img, result['key_points'], None, color=(0, 255, 255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img3 = cv2.drawMatches(query_img_sift, None,result_img_sift, None, None, None)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    plt.imshow(img3), plt.xticks([]), plt.yticks([])
    plt.show()

    # Show knn matching points
    img3 = cv2.drawMatchesKnn(img, kps, result_img, result['key_points'],result['good_points'], None, matchColor=(0, 255, 255),flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    plt.imshow(img3)
    plt.show()

    # Show only inliers
    drawParams = dict(matchColor=(0, 255, 255),singlePointColor=None, matchesMask=result['mask'],flags=2)
    img3 = cv2.drawMatchesKnn(img, kps, result_img, result['key_points'],result['good_points'], None, **drawParams)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    plt.imshow(img3)
    plt.show()


# Create sift and index if argument is -cr 
if sys.argv[1] == "-cr":
    sift = cv2.xfeatures2d.SIFT_create()
    index = {}

    for imgPath in glob.glob("Database/**jpg"):
        img = cv2.imread(imgPath, 0)
        coef = min(1, 300 / img.shape[1])
        # Resize by coef
        img = cv2.resize(img, None, fx=coef, fy=coef,interpolation=cv2.INTER_AREA)

        (kps, descs) = sift.detectAndCompute(img, None)
        imgPath = imgPath.replace("Database/", '')
        index[imgPath] = pickleKeypoints(kps, descs)

        # Save index object
        with open("index.pickle", "wb") as file:
            pickle.dump(index, file)

# Search if argument is -cr or -r
if sys.argv[1] == "-cr" or sys.argv[1] == "-r":
    index = pickle.load(open("index.pickle", "rb"))
    for (k, v) in index.items():
        (kps, descs) = unpickleKeypoints(v)
        index[k] = (kps, descs)
    search(sys.argv[2], index)
# Console examples
# python search.py -cr hw7_poster_1.jpg
# python search.py -r hw7_poster_2.jpg
# python search.py -r hw7_poster_3.jpg