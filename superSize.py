"""
Author:     Cody Hawkins
Date:       4/6/2021
Class:      6420
Desc:       K-means cluster analysis to find chosen
            and super impose it onto the original
            map image.
"""
import cv2 as cv
import numpy as np


def find_roi(points, image):
    # helper function to help shift image onto screen
    positions = []

    x, y, w, h = points[0]

    y = y // 2

    y_start, y_end = y + (h // 2), y + (h // 2) + (h * 2)
    x_start, x_end = x - (w // 2), x - (w // 2) + (w * 2)

    if y_end > image.shape[0]:
        y_start = y_start - (y_end - (image.shape[0]))
        y_end = y_end - (y_end - image.shape[0])
    if (y * 2) - (h // 2) < 0:
        y_start = y_start + (y - ( h // 2))
        y_end = y_end + (y - (h // 2))
    if x_end > image.shape[1]:
        x_start = x_start - (image.shape[1] + 1)
        x_end = x_end - (image.shape[1] + 1)
    if x_start < 0:
        x_end = x_end + ((-1 * (x_start)) + 1)
        x_start = x_start + ((-1 * (x_start)) + 1)

    positions.append((y_start, y_end, x_start, x_end))

    return positions


def k_mean(image):

    z = image.reshape((-1, 3))

    z = np.float32(z)

    criteria = (cv.TERM_CRITERIA_EPS, 10, 1.0)
    # 48 states plus black background
    k = 21

    ret, labels, centers = cv.kmeans(z, k, None, criteria, 10, cv.KMEANS_PP_CENTERS)

    centers = np.uint8(centers)

    labels = labels.flatten()

    return centers, labels


