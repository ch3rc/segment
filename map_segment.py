"""
Author:     Cody Hawkins
Date:       4/5/2021
Class:      6420
Desc:       Allow user to chose a state to super impose
            on map image. Use k-means clustering to
            distinguish colors of chose states
"""

import os
import sys
import getopt
import cv2 as cv
import numpy as np
from superSize import k_mean, find_roi

ix, iy = -1, -1
copy_img = None
points = []


def usage():
    print("-M or --manual:  Manually choose picture from file explorer rather than command line")
    print("User specifies state they want super imposed on the map by left clicking on a state")
    print("The program will use k-means clustering to find the centroid and super imposing the state")
    print("When finished press (q) to quit")


def resize(image):
    r, c = image.shape[:2]
    scale = 50
    width = int(c * scale / 100)
    height = int(r * scale / 100)
    dim = (width, height)
    return cv.resize(image, dim, interpolation=cv.INTER_CUBIC)


def pad(image):
    top = 8
    bottom = 8
    left = 8
    right = 8
    value = [0, 0, 0]
    dst = cv.copyMakeBorder(image.copy(), top, bottom, left, right, cv.BORDER_CONSTANT, None, value)
    return dst


def click_event(event, x, y, flags, param):
    global ix, iy, copy_img, points

    if event == cv.EVENT_LBUTTONDOWN:
        ix, iy = x, y
        color = copy_img[y][x]
        cluster = 0

        # search for the cluster with correct color
        for i, cent in enumerate(param[0]):
            count = 0
            for j in range(len(cent)):
                if color[j] == cent[j]:
                    count += 1
                if count == 3:
                    cluster = i

        result = copy_img.reshape((-1, 3))
        result[param[1]]
        result = result.reshape(copy_img.shape)

        # create mask which with black background and only states of same color
        masked = np.copy(param[2])
        masked = masked.reshape((-1, 3))
        masked[param[1] != cluster] = [0, 0, 0]
        masked = masked.reshape(copy_img.shape)

        res_g = cv.cvtColor(masked, cv.COLOR_BGR2GRAY)

        # get outlines of states
        canny_output = cv.Canny(res_g, 127, 255)

        _, canny_thresh = cv.threshold(canny_output, 127, 255, 0)

        contours, hierarchy = cv.findContours(canny_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # find our bounding boxes of state of choice
        pos = []
        ROI = None
        for c in contours:
            x1, y1, w, h = cv.boundingRect(c)

            if x1 < x < (x1 + w) and y1 < y < (y1 + h):
                pos.append((x1, y1, w, h))
                ROI = masked[y1: y1 + h, x1: x1 + w, :]
                break

        # double state size
        src = np.array([[0, 0], [ROI.shape[1] - 1, 0], [0, ROI.shape[0] - 1]]).astype(np.float32)
        dst = np.array([[0, 0], [(ROI.shape[1] * 2) - 1, 0], [0, (ROI.shape[0] * 2) - 1]]).astype(np.float32)
        warped_mat = cv.getAffineTransform(src, dst)
        big_roi = cv.warpAffine(ROI, warped_mat, (ROI.shape[1] * 2, ROI.shape[0] * 2))

        p = find_roi(pos, param[2])

        x0, x1, y0, y1 = p[0]

        # create mask of area and place in original image
        area = result[x0: x1, y0: y1]

        gray_roi = cv.cvtColor(big_roi, cv.COLOR_BGR2GRAY)

        ret, mask = cv.threshold(gray_roi, 10, 255, cv.THRESH_BINARY)

        roi_mask_inv = cv.bitwise_not(mask)

        result_roi = cv.bitwise_and(area, area, mask=roi_mask_inv)

        roi_area = cv.bitwise_and(big_roi, big_roi, mask=mask)

        dest = cv.add(result_roi, roi_area)

        result[x0: x1, y0: y1] = dest

        copy_img = result
        cv.imshow("usa", copy_img)

    elif event == cv.EVENT_LBUTTONUP:
        # reset image for next click
        copy_img = param[2]


def run_system(image):
    global copy_img, points
    img = pad(image)
    new_img = resize(img)
    centers, label = k_mean(new_img)
    result = centers[label]
    result = result.reshape(new_img.shape)
    copy_img = np.copy(result)
    while True:
        try:
            cv.imshow("usa", copy_img)
            cv.setMouseCallback("usa", click_event, (centers, label, result.copy()))
            k = cv.waitKey(33)
            if k == 113:
                break
        except cv.error as err:
            pass
    cv.destroyAllWindows()

    # superImpose(img, points.copy())


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hM", ["help", "manual"])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(1)

    image = None
    path = "C:\\Users\\codyh\\Desktop\\Test Pics"
    manual = False

    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit(1)
        elif o in ("-M", "--manual"):
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            image = filedialog.askopenfilename()
            manual = True
        else:
            assert False, "Unknown Option!"

    if not manual:

        if len(args) == 0:
            print("Please provide an input image!")
            usage()
            sys.exit(1)

        if len(args) == 1:
            image = os.path.join(path, args[0])
            try:
                img = cv.imread(image)
                if img is not None:
                    run_system(img)
            except cv.error as err:
                print(err)
                usage()
                sys.exit(1)

        if len(args) > 1:

            print("Too many inputs! Only need one input image.")
            usage()
            sys.exit(1)
    else:
        try:
            img = cv.imread(image)
            if img is not None:
                run_system(img)
        except cv.error as err:
            print(err)
            usage()
            sys.exit(1)


if __name__ == "__main__":
    main()