import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import bar

from skimage.exposure import histogram
from skimage.color import rgb2gray, rgb2hsv
from skimage.draw import rectangle
from skimage.measure import find_contours
from skimage.morphology import binary_closing, binary_dilation, binary_erosion, binary_opening, disk
from skimage.io import imread, imshow, imsave
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from BaseComponent import *

import sys
from os import path, makedirs


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


def show_images_rows(images, titles=None, windowTitle=None):
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    if windowTitle:
        fig.canvas.set_window_title(windowTitle)
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


def show_images_columns(images, titles=None, windowTitle=None):
    n_ims = len(images)
    if not titles:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    if windowTitle:
        fig.canvas.set_window_title(windowTitle)
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(2, round(n_ims/2), n)
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

        # Filter out barlines
        if ar >= 0.2:
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
def close_notes(image, staffDim):
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
    linePixels = np.argwhere(linesOnly)

    def is_staff_line(r, c):
        return np.where((linePixels == (r, c)).all(axis=1))[0].size > 0

    def is_connected_to_note(r, c, direction):
        # End of run
        if not image[r, c]:
            return False

        # Run continues into a note
        if not is_staff_line(r, c):
            return True

        return is_connected_to_note(r + direction, c, direction)

    for r, c in linePixels:
        # No connectivity to note
        if not is_connected_to_note(r, c, 1) and not is_connected_to_note(r, c, -1):
            clean[r, c] = False

    return clean


def extract_heads(image, staffDim):
    # Extract solid heads
    SIZE = 3 * staffDim[0]
    SE_disk = disk(SIZE)
    solidHeads = binary_opening(image, SE_disk)
    solidHeads = keep_elements_in_ar_range(solidHeads, 0.9, 2)

    # Join flags with solid heads
    # Close hollow heads
    heads = np.where(solidHeads, False, image)
    SE_disk = disk(SIZE+1)
    heads = binary_closing(heads, SE_disk)
    heads = binary_opening(heads, SE_disk)
    heads = keep_elements_in_ar_range(heads, 0.9, 1.75)

    # Mask = remove_noise(closed hollow heads + solid heads)
    mask = binary_opening(heads | solidHeads)

    return mask


def keep_elements_in_ar_range(image, lower, upper):
    try:
        contours = find_contours(image, 0.8)
    except:
        return image

    filtered = np.copy(image)
    for c in contours:
        xValues = np.round(c[:, 1]).astype(int)
        yValues = np.round(c[:, 0]).astype(int)
        ar = (xValues.max() - xValues.min()) / (yValues.max() - yValues.min())

        if ar < lower or ar > upper:
            rr, cc = rectangle(start=(yValues.min(), xValues.min()), end=(
                yValues.max(), xValues.max()), shape=filtered.shape)
            filtered[rr, cc] = 0

    return filtered


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
            xpos.append(c.x + max(0, i-2))
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


def divide_beams(baseComponents, image, staffDim):
    allBaseComponents = []
    for idx, v in enumerate(baseComponents):
        slicedImage = image[v.y:v.y+v.height, v.x:v.x+v.width]
        heads = extract_heads(slicedImage, staffDim)
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
        if v.ar > 0.27:
            cleanBaseComponents.append(v)

    return cleanBaseComponents


def sanitize_sheet(image):
    linesOnly, staffDim = extract_staff_lines(image)

    # Image - Lines
    removedLines = remove_staff_lines(image, linesOnly, staffDim)
    closedNotes = close_notes(removedLines, staffDim)

    # Clef removal
    firstRun = get_first_run(closedNotes)
    closedNotes[:, firstRun] = 0

    # This step automatically removes barlines
    masked, mask = mask_image(closedNotes, removedLines)

    return masked, closedNotes, staffDim


def save_segments(segments):
    for i, seg in enumerate(segments):
        segment = seg.astype(np.uint8) * 255
        if not path.exists('samples'):
            makedirs('samples')
        imsave(f'samples/beam{i}.jpg', segment)
