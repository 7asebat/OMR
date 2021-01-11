from skimage.transform import hough_line, rotate
from skimage.io import imread, imsave
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray, rgba2rgb
from matplotlib import pyplot as plt
import re
import sys
import cv2
import numpy as np
from Display import show_images


def should_rotate_edge_image(edge_image):
    h, theta, _ = hough_line(edge_image)
    max_index = np.unravel_index(h.argmax(), h.shape)
    degreePeak = np.rad2deg(theta[max_index[1]])
    if(abs(degreePeak) > 88 and abs(degreePeak) < 92):
        return False
    return True


def biggest_contour(contours, min_area):
    max_area = 0
    biggest_n = 0
    approx_contour = None
    for n, i in enumerate(contours):
        area = cv2.contourArea(i)

        # if area > min_area/20:
        peri = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.2*peri, True)
        if area > max_area:
            biggest = approx
            max_area = area
            biggest_n = n
            approx_contour = approx

    return biggest_n, approx_contour


def order_points(pts):
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def simplify_contour(contour, n_corners=4):
    n_iter, max_iter = 0, 100
    lb, ub = 0., 1.

    while True:
        n_iter += 1
        if n_iter > max_iter:
            return contour

        k = (lb + ub)/2.
        eps = k*cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)

        if len(approx) > n_corners:
            lb = (lb + ub)/2.
        elif len(approx) < n_corners:
            ub = (lb + ub)/2.
        else:
            return approx


def transformation(image):
    image = image.copy()
    # originalImage = np.copy(image)
    image = cv2.copyMakeBorder(
        image, 250, 250, 250, 250, cv2.BORDER_REPLICATE)
    image_size = image.size

    ## PREPROCESSING ##
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    _, threshold = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # lines_edges = cv2.Canny(threshold, 50, 150, apertureSize=7)

    dilate = cv2.dilate(threshold, np.ones(
        (9, 9), np.uint8), iterations=8)

    edges = cv2.Canny(dilate, 50, 150, apertureSize=7)

    ## FIND CONTOUR ##
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # img_with_contours = np.copy(image)
    # cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 3)
    simplified_contours = []
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        simplified_contours.append(cv2.approxPolyDP(hull,
                                                    0.0001*cv2.arcLength(hull, True), True))
    # img_with_simplified_contours = np.copy(image)
    # cv2.drawContours(img_with_simplified_contours,
    #                  simplified_contours, -1, (255, 0, 0), 3)

    # GET BIGGEST CONTOUR ##

    simplified_contours = np.array(simplified_contours)
    biggest_n, approx_contour = biggest_contour(
        simplified_contours, image_size)

    # contouredImage = np.copy(image)
    # cv2.drawContours(
    #     contouredImage, simplified_contours, biggest_n, (0, 0, 255), 3)

    # quad_img = np.copy(image)
    # quadContour = simplify_contour(simplified_contours[biggest_n], 4)
    # img_with_quadContour = cv2.drawContours(
    #     quad_img, [quadContour], 0, (0, 0, 255), 2)

    # rect_img = np.copy(image)

    rect = cv2.minAreaRect(simplified_contours[biggest_n])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # img_with_rect = cv2.drawContours(rect_img, [box], 0, (0, 0, 255), 2)

    dst = image
    dst = four_point_transform(image, box)

    croppedImage = np.copy(dst)

    return croppedImage

# **Sharpen the image using Kernel Sharpening Technique**


def final_image(rotated):
    kernel_sharpening = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])

    sharpened = cv2.filter2D(rotated, -1, kernel_sharpening)

    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    _, threshold = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return threshold


def fix_rotation(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=7)
    h, theta, d = hough_line(edges)

    max_index = np.unravel_index(h.argmax(), h.shape)
    degreePeak = np.rad2deg(theta[max_index[1]])

    rotationAngle = degreePeak + 90
    if(rotationAngle < 45):
        return image

    rotatedImg = rotate(image, rotationAngle, cval=1,
                        resize=True)
    rotatedImg = (rotatedImg * 255).astype(np.uint8)

    return rotatedImg


def should_rotate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

   # blur = cv2.GaussianBlur(gray, (5, 5), 2)
    _, threshold = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    lines_edges = cv2.Canny(threshold, 50, 150, apertureSize=7)

    h, theta, _ = hough_line(lines_edges)
    max_index = np.unravel_index(h.argmax(), h.shape)
    degreePeak = np.rad2deg(theta[max_index[1]])

    return not (88 < abs(degreePeak) < 92)


def fix_orientation(image):
    cleaned_image = image
    blurred_threshold = transformation(image)
    cleaned_image = final_image(blurred_threshold)
    fixed_rotation_image = fix_rotation(cleaned_image)
    return fixed_rotation_image


def read_and_preprocess_image(path):
    image = cv2.imread(path)
    useAugmented = should_rotate(image)
    if useAugmented:
        image = fix_orientation(image)
        image = ~image
    else:
        if len(image.shape) == 3:
            if image.shape[2] > 3:
                image = rgba2rgb(image)
            image = (rgb2gray(image) * 255).astype(np.uint8)

        image = image < threshold_otsu(image)

    # @note Here we remove the first row of the image
    #       until we trim the image
    return image[1:, :], useAugmented
