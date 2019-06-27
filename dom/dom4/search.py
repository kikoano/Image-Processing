import cv2
import pickle

## Import zernikeMoments from parent file
import os
import sys
module_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(module_dir, '../dom3/'))
from zernikeMoments import zernikeMoments
##

from scipy.spatial import distance as dist
from matplotlib import pyplot as plt

# Import index
index = pickle.load(open('../dom3/index.pickle', 'rb'))

# Search
def search(imgPath):
    img = cv2.imread(imgPath, 0)
    moments = zernikeMoments(img)
    res = {}

    for (k, v) in index.items():
        # Use correlation to calculate distance
        res[k] = dist.correlation(v, moments)
    res = sorted([(v, k) for (k, v) in res.items()])
    # Get 6 from the sort order
    return res[:6]

# Get image source argument
srcImg = sys.argv[1]
# Search for best 6 results
result = search(srcImg)
# Console examples
# python search.py ../dom3/query/12034.jpg
# python search.py ../dom3/query/14729.jpg
# python search.py ../dom3/query/11297.jpg

# Offset
off = 231
for i in range(6):
    plt.subplot(off+i)
    imgPath = "../dom3/" + result[i][1]
    img = cv2.imread(imgPath, cv2.IMREAD_COLOR)

    # Convert colors for show
    imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(imgRgb), plt.xticks([]), plt.yticks([])

    plt.xlabel('Distance: {0:.6f}'.format(result[i][0]))
    # Name of result image
    plt.title(imgPath.split("\\")[-1])
plt.show()