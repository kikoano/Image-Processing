import cv2
import numpy as np
from matplotlib import pyplot as plt


def compass(m1, m2, m3=0, m4=0, m5=0, m6=0, m7=0, m8=0, figure=1):
    img = cv2.imread('image.png')
    img = np.float32(img)

    o1 = np.array((m1), dtype="float32")
    filteredImg1 = cv2.filter2D(img, -1, o1)
    filteredImg1 = abs(filteredImg1)
    filteredImg1 = filteredImg1 / np.amax(filteredImg1[:])

    o2 = np.array((m2), dtype="float32")
    filteredImg2 = cv2.filter2D(img, -1, o2)
    filteredImg2 = abs(filteredImg2)
    filteredImg2 = filteredImg2 / np.amax(filteredImg2[:])

    filteredImg3 = np.zeros(1)
    filteredImg4 = np.zeros(1)
    filteredImg5 = np.zeros(1)
    filteredImg6 = np.zeros(1)
    filteredImg7 = np.zeros(1)
    filteredImg8 = np.zeros(1)

    if m3 != 0:
        o3 = np.array((m3), dtype="float32")
        filteredImg3 = cv2.filter2D(img, -1, o3)
        filteredImg3 = abs(filteredImg3)
        filteredImg3 = filteredImg3 / np.amax(filteredImg3[:])

    if m4 != 0:
        o4 = np.array((m4), dtype="float32")
        filteredImg4 = cv2.filter2D(img, -1, o4)
        filteredImg4 = abs(filteredImg4)
        filteredImg4 = filteredImg4 / np.amax(filteredImg4[:])

    if m5 != 0:
        o5 = np.array((m5), dtype="float32")
        filteredImg5 = cv2.filter2D(img, -1, o5)
        filteredImg5 = abs(filteredImg5)
        filteredImg5 = filteredImg5 / np.amax(filteredImg5[:])

    if m6 != 0:
        o6 = np.array((m6), dtype="float32")
        filteredImg6 = cv2.filter2D(img, -1, o6)
        filteredImg6 = abs(filteredImg6)
        filteredImg6 = filteredImg6 / np.amax(filteredImg6[:])

    if m7 != 0:
        o7 = np.array((m7), dtype="float32")
        filteredImg7 = cv2.filter2D(img, -1, o7)
        filteredImg7 = abs(filteredImg7)
        filteredImg7 = filteredImg7 / np.amax(filteredImg7[:])

    if m8 != 0:
        o8 = np.array((m8), dtype="float32")
        filteredImg8 = cv2.filter2D(img, -1, o8)
        filteredImg8 = abs(filteredImg8)
        filteredImg8 = filteredImg8 / np.amax(filteredImg8[:])

    compasOperators = [filteredImg1, filteredImg2, filteredImg3,
                       filteredImg4, filteredImg5, filteredImg6, filteredImg7, filteredImg8]
    compasOperatorsSumMask = [filteredImg1.sum(), filteredImg2.sum(), filteredImg3.sum(
    ), filteredImg4.sum(), filteredImg5.sum(), filteredImg6.sum(), filteredImg7.sum(), filteredImg8.sum()]
    maxRes = compasOperators[np.argmax(compasOperatorsSumMask)]

    plt.figure(figure)

    plt.subplot(331)
    plt.imshow(filteredImg1)

    plt.subplot(332)
    plt.imshow(filteredImg2)

    if m3 != 0:
        plt.subplot(333)
        plt.imshow(filteredImg3)

    if m4 != 0:
        plt.subplot(334)
        plt.imshow(filteredImg4)

    plt.subplot(335)
    plt.title("Maximum response")
    plt.imshow(maxRes)

    if m5 != 0:
        plt.subplot(336)
        plt.imshow(filteredImg5)

    if m6 != 0:
        plt.subplot(337)
        plt.imshow(filteredImg6)

    if m7 != 0:
        plt.subplot(338)
        plt.imshow(filteredImg7)

    if m8 != 0:
        plt.subplot(339)
        plt.imshow(filteredImg8)

    plt.imsave("compass"+str(figure)+".png", maxRes)


compass([[1, 1, 0], [1, 0, -1], [0, -1, -1]],
        [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
        [[0, 1, 1], [-1, 0, 1], [-1, -1, 0]],
        [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
        [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
        [[0, -1, -1], [1, 0, -1], [1, 1, 0]],
        [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
        [[-1, -1, 0], [-1, 0, 1], [0, 1, 1]], figure=1)

compass([[1, 1, 1], [1, -2, 1], [-1, -1, -1]],
        [[5, 5, 5], [-3, 0, -3], [-3, -3, -3]],
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], figure=2)

plt.show()
