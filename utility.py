import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import bar

from skimage.exposure import histogram
from skimage.color import rgb2gray, rgb2hsv
from skimage.draw import rectangle
from skimage.measure import find_contours
from skimage.morphology import binary_closing, binary_dilation, binary_erosion, binary_opening
from skimage.io import imread, imshow
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from BaseComponent import *

import sys


def show_images(images, titles=None):
    # This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        a.xaxis.set_ticks([])
        a.yaxis.set_ticks([])
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    fig.tight_layout()
    plt.show()


def show_images_2(images, titles=None):
    # This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(round(n_ims/2), 2, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        a.xaxis.set_ticks([])
        a.yaxis.set_ticks([])
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    fig.tight_layout()
    plt.show()


def showHist(image):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imageHist = histogram(image, nbins=256)

    bar(imageHist[1].astype(np.uint8), imageHist[0], width=0.8, align='center')


# When provided with the correct format of the list of bounding_boxes, this section will set all pixels inside boxes in image_with_boxes
def set_pixels(image, bounding_boxes):
    image_with_boxes = np.copy(image)
    for box in bounding_boxes:
        [Xmin, Xmax, Ymin, Ymax] = box
        rr, cc = rectangle(start=(Ymin, Xmin), end=(
            Ymax, Xmax), shape=image.shape)
        image_with_boxes[rr, cc] = 1  # set color white
    return image_with_boxes


def get_bounding_boxes(image):
    boundingBoxes = []
    contours = find_contours(image, 0.8)
    for c in contours:
        xValues = np.round(c[:, 1]).astype(int)
        yValues = np.round(c[:, 0]).astype(int)
        ar = (xValues.max() - xValues.min()) / (yValues.max() - yValues.min())

        if 0.25 <= ar:
            boundingBoxes.append(
                [xValues.min(), xValues.max(), yValues.min(), yValues.max()])

    return boundingBoxes


def get_vertical_histogram(image):
    return np.array([image[:, col].sum() for col in range(image.shape[1])])


def get_vertical_histogram_image(hist, shape):
    histimage = np.zeros(shape)
    for c, height in enumerate(hist):
        if height:
            histimage[-height:, c] = True

    return histimage


def get_horizontal_histogram(image):
    return np.array([image[row].sum() for row in range(image.shape[0])])


def get_horizontal_histogram_image(hist, shape):
    histimage = np.zeros(shape)
    for r, width in enumerate(hist):
        if width:
            histimage[r, :width] = True

    return histimage


def get_staff_line_dimensions(image):
    hHist = get_horizontal_histogram(image)
    length = hHist.max()
    lengthThreshold = 0.85 * length
    linesOnly = np.copy(image)
    for ir, _ in enumerate(linesOnly):
        if hHist[ir] < lengthThreshold:
            linesOnly[ir] = False


def extract_staff_lines(image):
    # Staff lines using a horizontal histogram
    # > Get staff line length
    # > Threshold image based on said length
    hHist = get_horizontal_histogram(image)
    lengthThreshold = 0.85 * hHist.max()
    linesOnly = np.copy(image)
    for ir, _ in enumerate(linesOnly):
        if hHist[ir] < lengthThreshold:
            linesOnly[ir] = False

    # > Get staff line width
    length = hHist.max()
    run = 0
    runFreq = {}
    lineHist = (hHist >= lengthThreshold) * length
    for row in lineHist:
        if row:
            run += 1
        elif run:
            if run in runFreq:
                runFreq[run] += 1
            else:
                runFreq[run] = 1
            run = 0
    width = max(runFreq, key=runFreq.get)

    return linesOnly, (width, length)


# Connects notes in an image with removed lines
def connect_notes(image, staffDim):
    SIZE = 5 * (staffDim[0]+1)
    SE_notes = np.ones((SIZE, 1))
    connectedNotes = binary_closing(image, SE_notes)  # Connect vertically
    return connectedNotes


def mask_image(connectedimage, image):
    mask = set_pixels(np.zeros(connectedimage.shape),
                      get_bounding_boxes(connectedimage))
    return np.where(mask, image, False), mask


def remove_non_vertical_protrusions(image, staffDim):
    # Remove non-vertical protrusions (to remove staff lines)
    # TODO: Use Hit-and-miss to remove lines instead
    SIZE = staffDim[0]+1
    SE_vertical = np.ones((SIZE, 1))
    return binary_opening(image, SE_vertical)  # Keep vertical elements


def remove_staff_lines(image, linesOnly, staffDim):
    clean = np.copy(image)
    # Remove pixels from image that exist in linesOnly which are only connected to other lines
    linePixels = np.argwhere(linesOnly)
    for r, c in linePixels:
        if not image[r + 1, c] or not image[r - 1, c]:
            clean[r, c] = False

    return clean


def isolate_heads(image, staffDim):
    # Isolate heads
    SIZE = 5*staffDim[0] + 1
    SE_box = np.ones((SIZE, SIZE))
    return binary_opening(image, SE_box)


def extract_heads(image):
    SIZE = 11
    SE_box = np.ones((SIZE, SIZE))
    heads = image
    heads = binary_opening(heads, SE_box)

    try:
        contours = find_contours(heads, 0.8)
    except:
        return heads

    for c in contours:
        xValues = np.round(c[:, 1]).astype(int)
        yValues = np.round(c[:, 0]).astype(int)
        ar = (xValues.max() - xValues.min()) / (yValues.max() - yValues.min())

        if ar > 2:
            rr, cc = rectangle(start=(yValues.min(), xValues.min()), end=(
                yValues.max(), xValues.max()), shape=image.shape)
            heads[rr, cc] = 0

    return heads


def divide_segment(image, vHist):
    xpos = []
    for i in range(1, len(vHist)):
        if(vHist[i] == True and vHist[i-1] == False):
            xpos.append(max(0, i-1))
    xpos.append(image.shape[1])
    divisions = []
    for i in range(len(xpos)-1):
        divisions.append(image[:, xpos[i]:xpos[i+1]])

    return divisions


def divide_component(c, vHist):
    xpos = []
    for i in range(1, len(vHist)):
        if(vHist[i] == True and vHist[i-1] == False):
            xpos.append(c.x + max(0, i-1))
    xpos.append(c.x + c.width)
    divisions = []
    for i in range(len(xpos)-1):
        newDivision = BaseComponent([xpos[i], xpos[i+1], c.y, c.y+c.height])
        divisions.append(newDivision)

    return divisions


def get_number_of_heads(vHist):
    numHeads = 0
    for i in range(1, len(vHist)):
        if(vHist[i] == True and vHist[i-1] == False):
            numHeads += 1
    return numHeads


def slice_image(image, boundingBoxes):
    slicedImage = []
    for box in boundingBoxes:
        [Xmin, Xmax, Ymin, Ymax] = box
        slicedImage.append(image[Ymin:Ymax, Xmin:Xmax])
    return slicedImage


def divide_beams(baseComponents, image):
    allBaseComponents = []
    for idx, v in enumerate(baseComponents):
        slicedImage = image[v.y:v.y+v.height, v.x:v.x+v.width]
        heads = extract_heads(slicedImage)
        vHist = get_vertical_histogram(heads) > 0
        numHeads = get_number_of_heads(vHist)
        if numHeads > 1:
            components = divide_component(v, vHist)
            allBaseComponents = allBaseComponents + components
        else:
            allBaseComponents.append(v)
    return allBaseComponents


def get_first_run(image):
    vHist = get_vertical_histogram(image)
    run = [-1, -1]
    for p, v in enumerate(vHist):
        if run[0] < 0:
            if v:
                run[0] = p

        else:
            if not v:
                run[1] = p
                break

    return slice(run[0], run[1])


def get_base_components(boundingBoxes):
    baseComponents = []
    for box in boundingBoxes:
        component = BaseComponent(box)
        baseComponents.append(component)
    return baseComponents


def remove_vertical_bar_components(baseComponents):
    cleanBaseComponents = []
    for v in baseComponents:
        if v.get_ar() > 0.27:
            cleanBaseComponents.append(v)

    return cleanBaseComponents
