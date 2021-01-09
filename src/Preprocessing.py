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
    h, theta, d = hough_line(edge_image)
    max_index = np.unravel_index(h.argmax(), h.shape)
    degreePeak = np.rad2deg(theta[max_index[1]])
    if(abs(degreePeak) > 88 and abs(degreePeak) < 92):
        return False
    return True


def biggest_contour(contours, min_area):
    # ## **Find the Biggest Contour**
    # **Note: We made sure the minimum contour is bigger than 1/10 size of the whole picture. This helps in removing very small contours (noise) from our dataset**
    biggest = None
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
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


# ## Find the exact (x,y) coordinates of the biggest contour and crop it out


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def simplify_contour(contour, n_corners=4):
    '''
    Binary searches best `epsilon` value to force contour 
        approximation contain exactly `n_corners` points.

    :param contour: OpenCV2 contour.
    :param n_corners: Number of corners (points) the contour must contain.

    :returns: Simplified contour in successful case. Otherwise returns initial contour.
    '''
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


def simplify_contour_2(c):
    pass


def show_image(img):
    cv2.imshow("Image", img)
    # press q or Esc to close
    cv2.waitKey(0)


def fix_skewing(im, max_skew=10):
    im_gs = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gs = cv2.fastNlMeansDenoising(im_gs, h=3)
    height, width = im_gs.shape

    # im = (im * 255).astype(np.uint8)

    # Detect lines in this image. Parameters here mostly arrived at by trial and error.
    lines = cv2.HoughLinesP(
        im_gs, 1, np.pi / 180, 200, minLineLength=width / 12, maxLineGap=width / 150
    )

    # Collect the angles of these lines (in radians)
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angles.append(np.arctan2(y2 - y1, x2 - x1))

    # If the majority of our lines are vertical, this is probably a landscape image
    landscape = np.sum(
        [abs(angle) > np.pi / 4 for angle in angles]) > len(angles) / 2

    # Filter the angles to remove outliers based on max_skew
    if landscape:
        angles = [
            angle
            for angle in angles
            if np.deg2rad(90 - max_skew) < abs(angle) < np.deg2rad(90 + max_skew)
        ]
    else:
        angles = [angle for angle in angles if abs(
            angle) < np.deg2rad(max_skew)]

    if len(angles) < 5:
        # Insufficient data to deskew
        return im

    # Average the angles to a degree offset
    angle_deg = np.rad2deg(np.median(angles))

    # If this is landscape image, rotate the entire canvas appropriately
    if landscape:
        if angle_deg < 0:
            im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
            angle_deg += 90
        elif angle_deg > 0:
            im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
            angle_deg -= 90

    # Rotate the image by the residual offset
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle_deg, 1)
    im = cv2.warpAffine(im, M, (width, height),
                        borderMode=cv2.BORDER_REPLICATE)

    # show_images([im])
    return im


def transformation(image):
    image = image.copy()
    originalImage = np.copy(image)
    image = cv2.copyMakeBorder(
        image, 500, 500, 500, 500, cv2.BORDER_REPLICATE)
    image_size = image.size

    ## PREPROCESSING ##
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    _, threshold = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    lines_edges = cv2.Canny(threshold, 50, 150, apertureSize=7)

    if not should_rotate_edge_image(lines_edges):
        print('Should not process')
        return originalImage

    dilate = cv2.dilate(threshold, np.ones(
        (9, 9), np.uint8), iterations=8)

    edges = cv2.Canny(dilate, 50, 150, apertureSize=7)
    # show_images([image, threshold, dilate, edges], [
    #             "Original", "Threshold", "Dilated", "Edges"])

    ## FIND CONTOUR ##
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_with_contours = np.copy(image)
    cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 3)
    simplified_contours = []
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        simplified_contours.append(cv2.approxPolyDP(hull,
                                                    0.0001*cv2.arcLength(hull, True), True))
    img_with_simplified_contours = np.copy(image)
    cv2.drawContours(img_with_simplified_contours,
                     simplified_contours, -1, (255, 0, 0), 3)

    # GET BIGGEST CONTOUR ##

    simplified_contours = np.array(simplified_contours)
    biggest_n, approx_contour = biggest_contour(
        simplified_contours, image_size)

    contouredImage = np.copy(image)
    cv2.drawContours(
        contouredImage, simplified_contours, biggest_n, (0, 0, 255), 3)

    quad_img = np.copy(image)
    quadContour = simplify_contour(simplified_contours[biggest_n], 4)
    img_with_quadContour = cv2.drawContours(
        quad_img, [quadContour], 0, (0, 0, 255), 2)

    rect_img = np.copy(image)

    rect = cv2.minAreaRect(simplified_contours[biggest_n])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    img_with_rect = cv2.drawContours(rect_img, [box], 0, (0, 0, 255), 2)

    # show_images([image, img_with_contours, contouredImage, img_with_rect], [
    #             "Image", "Contour", "Simplified Contour", "Min Area Rect"])

    dst = image
    dst = four_point_transform(image, box)

    croppedImage = np.copy(dst)

    return croppedImage


# **Increase the brightness of the image by playing with the "V" value (from HSV)**

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


# **Sharpen the image using Kernel Sharpening Technique**


def final_image(rotated):
    # Create our shapening kernel, it must equal to one eventually
    kernel_sharpening = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])
    # applying the sharpening kernel to the input image & displaying it.
    sharpened = cv2.filter2D(rotated, -1, kernel_sharpening)
    # sharpened = increase_brightness(sharpened, 30)
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    _, threshold = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # show_images([threshold])

    return threshold


def fix_rotation(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=7)
    h, theta, d = hough_line(edges)
    # peaks = cv2.hough_line_peaks(h, theta, d)
    max_index = np.unravel_index(h.argmax(), h.shape)
    degreePeak = np.rad2deg(theta[max_index[1]])

    rotationAngle = degreePeak + 90
    if(rotationAngle < 45):
        return image

    rotatedImg = rotate(image, rotationAngle, cval=1, resize=True)
    # show_images([rotatedImg])
    return rotatedImg


def should_rotate(image):
    print(image.shape, image.dtype)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    lines_edges = cv2.Canny(threshold, 50, 150, apertureSize=7)

    h, theta, _ = hough_line(lines_edges)
    max_index = np.unravel_index(h.argmax(), h.shape)
    degreePeak = np.rad2deg(theta[max_index[1]])

    return not (88 < abs(degreePeak) < 92)


def fix_orientation(image):
    # 1. Pass the image through the transformation function to crop out the biggest contour
    # 2. Brighten & Sharpen the image to get a final cleaned image
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
    else:
        if len(image.shape) > 2:
            if image.shape[2] > 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
                # image = rgba2rgb(image)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # image = (rgb2gray(image) * 255).astype(np.uint8)

            image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # image = image < threshold_otsu(image)

    # @note Here we remove the first row of the image
    #       until we trim the image
    return image[1:, :], useAugmented
