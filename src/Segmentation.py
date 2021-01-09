from cv2 import copyMakeBorder, BORDER_CONSTANT, getStructuringElement, MORPH_ELLIPSE, morphologyEx, MORPH_OPEN, MORPH_ERODE, MORPH_CLOSE
import Utility
from scipy.ndimage.interpolation import rotate
from skimage.morphology import binary_closing, binary_dilation, binary_erosion, binary_opening, disk, selem
import numpy as np


def extract_heads(image, staffDim, filterAR=True):
    staffSpacing = staffDim[2]
    vHist = Utility.get_vertical_projection(image) > 0
    numHeads = get_number_of_heads(vHist)

    closedImage = binary_closing(image)
    if numHeads > 1:
        closedImage = binary_closing(image, np.ones((10, 10), dtype='bool'))

    # Extract solid heads
    # @note skimage sucks
    w = staffSpacing
    h = int(staffSpacing * 6/7)
    SE_ellipse = getStructuringElement(MORPH_ELLIPSE, (w, h))
    SE_ellipse = rotate(SE_ellipse, angle=30)

    solidHeads = morphologyEx(closedImage.astype(np.uint8),
                              MORPH_OPEN,
                              SE_ellipse,
                              borderType=BORDER_CONSTANT,
                              borderValue=0)

    if filterAR:
        solidHeads = Utility.keep_elements_in_ar_range(solidHeads, 0.9, 1.5)

    mask = binary_opening(solidHeads)

    return mask


def get_number_of_heads(vHist):
    numHeads = 0
    for i, _ in enumerate(vHist[:-1]):
        if not i and vHist[i]:
            numHeads += 1

        elif not vHist[i] and vHist[i + 1]:
            numHeads += 1

    # numHeads = 0
    # bw = False
    # for i, _ in enumerate(vHist[:-1]):
    #     if not bw and vHist[i] and not vHist[i+1]:
    #         numHeads += 1
    #         bw = False

    #     elif not vHist[i] and vHist[i + 1]:
    #         numHeads += 1
    #         bw = True

    return numHeads


def detect_chord(slc, staffDim):
    staffSpacing = staffDim[2]
    heads = np.copy(slc).astype(np.uint8)

    # Use an elliptical structuring element
    w = staffSpacing // 2
    h = int(staffSpacing * 5/6) // 2
    SE_ellipse = getStructuringElement(MORPH_ELLIPSE, (w, h))
    SE_ellipse = rotate(SE_ellipse, angle=30)

    # @note skimage sucks
    heads = morphologyEx(heads, MORPH_ERODE, SE_ellipse,
                         borderType=BORDER_CONSTANT, borderValue=0)
    heads = morphologyEx(heads, MORPH_OPEN, SE_ellipse,
                         borderType=BORDER_CONSTANT, borderValue=0)

    boundingBoxes = Utility.get_bounding_boxes(heads)
    numHeads = len(boundingBoxes)

    return numHeads > 1


def detect_art_dots(image, sanitized, staffDim):
    k = np.zeros((3, 3), dtype='uint8')
    k[3//2:3//2+1, :] = 1

    sanitized = binary_opening(sanitized, k)

    # Get base of components from boundingBoxes
    boxes = Utility.get_bounding_boxes(sanitized, 0.8, 1.35)

    min_area, max_area = (staffDim[2] // 4)**2, (staffDim[2] // 2)**2
    art_dots_img = np.zeros(image.shape, dtype='uint8')
    dotBoxes = []

    for xl, xh, yl, yh in boxes:
        slc = (slice(yl, yh), slice(xl, xh))
        area = (xh-xl)*(yh-yl)
        white = np.argwhere(sanitized[slc]).shape[0]
        ratio = white / area

        if area > min_area and area < max_area and ratio > 0.68:
            dotBoxes.append((xl, xh, yl, yh))
            art_dots_img[slc] = 1

    dotMask = binary_dilation(art_dots_img, np.ones((4, 4), dtype='uint8'))
    return dotMask, dotBoxes
