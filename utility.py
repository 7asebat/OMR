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
from scipy.signal import find_peaks
from BaseComponent import *

import sys
import os
import json
from glob import glob
from shutil import rmtree


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
        Xmin, Xmax, Ymin, Ymax = box
        rr, cc = rectangle(start=(Ymin, Xmin), end=(
            Ymax, Xmax), shape=image.shape)
        image_with_boxes[rr, cc] = 1  # set color white
    return image_with_boxes


def get_bounding_boxes(image, lower=0, upper=sys.maxsize):
    def is_subset(l, r):
        subset = l[0] >= r[0]
        subset &= l[1] <= r[1]
        subset &= l[2] >= r[2]
        subset &= l[3] <= r[3]
        return subset

    boundingBoxes = []
    contours = find_contours(image, 0.8)
    for c in contours:
        xValues = np.round(c[:, 1]).astype(int)
        yValues = np.round(c[:, 0]).astype(int)
        box = (xValues.min(), xValues.max(), yValues.min(), yValues.max())
        ar = (xValues.max() - xValues.min()) / (yValues.max() - yValues.min())

        # Filter out barlines
        # Keep larger bounding box
        # Filter overlapping bounding boxes
        insertNew = True
        if lower <= ar <= upper:
            for x in boundingBoxes:
                if is_subset(x, box):
                    boundingBoxes.remove(x)
                elif is_subset(box, x):
                    insertNew = False

            if insertNew:
                boundingBoxes.append(box)

    return boundingBoxes


def get_vertical_projection(image):
    return np.array([image[:, col].sum() for col in range(image.shape[1])])


def get_vertical_projection_image(hist, shape):
    histimage = np.zeros(shape)
    for c, height in enumerate(hist):
        if height:
            histimage[-height:, c] = True

    return histimage


def get_horizontal_projection(image):
    return np.array([image[row].sum() for row in range(image.shape[0])])


def get_horizontal_projection_image(hist, shape):
    histimage = np.zeros(shape)
    for r, width in enumerate(hist):
        if width:
            histimage[r, :width] = True

    return histimage


def extract_staff_lines(image):
    '''
    @return (LineImage, (width, length, spacing))
    '''
    # Staff lines using a horizontal histogram
    # > Get staff line length
    # > Threshold image based on said length
    linesOnly = np.copy(image)

    hHist = get_horizontal_projection(image)
    length = hHist.max()
    lengthThreshold = 0.85 * hHist.max()

    hHist[hHist < lengthThreshold] = 0
    for r, val in enumerate(hHist):
        if not val:
            linesOnly[r] = 0

    # > Get staff line width
    run = 0
    runFreq = {}
    for val in hHist:
        if val:
            run += 1
        elif run:
            if run in runFreq:
                runFreq[run] += 1
            else:
                runFreq[run] = 1
            run = 0
    width = max(runFreq, key=runFreq.get)

    # > Get staff line spacing
    # > Find the space between any two consecutive runs
    rows = []
    run = False
    spacing = -1
    for r, val in enumerate(hHist):
        if val:
            run = True

        elif run:
            rows.append(r)

            if (len(rows) > 1):
                spacing = rows[1] - rows[0]
                break

            run = 0

    return linesOnly, (width, length, spacing)


def close_notes(image, staffDim):
    '''
    Connects notes in an image with removed lines
    '''
    staffWidth = staffDim[0]
    SIZE = 5 * (staffWidth+1)
    SE_notes = np.ones((SIZE, 1))
    connectedNotes = binary_closing(image, SE_notes)  # Connect vertically
    return connectedNotes


def mask_image(connectedimage, image):
    '''
    @note Extract the low ar value
    '''
    boundingBoxes = get_bounding_boxes(connectedimage, 0.2)
    mask = set_pixels(np.zeros(connectedimage.shape), boundingBoxes)
    return np.where(mask, image, False), mask


def remove_staff_lines(image, linesOnly, staffDim):
    clean = np.copy(image)
    linePixels = np.argwhere(linesOnly)

    def is_connected_to_note(r, c, direction):
        # End of run
        if not image[r, c]:
            return False

        # Run continues into a note
        if not linesOnly[r, c]:
            return True

        return is_connected_to_note(r + direction, c, direction)

    for r, c in linePixels:
        # No connectivity to note
        if not is_connected_to_note(r, c, 1) and not is_connected_to_note(r, c, -1):
            clean[r, c] = False

    return clean


def extract_heads(image, staffDim):
    # Extract solid heads
    #staffWidth = staffDim[0]
    staffSpacing = staffDim[2]

    SIZE = (staffSpacing-3)//2
    SE_disk = disk(SIZE)

    vHist = get_vertical_projection(image) > 0
    numHeads = get_number_of_heads(vHist)

    closedImage = image
    if numHeads > 1:
        closedImage = binary_closing(image, np.ones((10, 10), dtype='bool'))

    solidHeads = binary_opening(closedImage, SE_disk)

    filteredSolidHeads = keep_elements_in_ar_range(solidHeads, 0.9, 1.5)
    heads = 0

    # @note Uncomment this section to include detection of hollow note heads
    # Join flags with solid heads
    # Close hollow heads
    # heads = np.where(solidHeads, False, image)
    # SE_disk = disk(SIZE+1)
    # heads = binary_closing(heads, SE_disk)
    # heads = binary_opening(heads, SE_disk)
    # heads = keep_elements_in_ar_range(heads, 0.9, 1.75)

    # SIZE = 4 * staffWidth
    # SE_disk = disk(SIZE)

    # Mask = remove_noise(closed hollow heads + solid heads)
    mask = binary_opening(heads | filteredSolidHeads)

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


def divide_component(c, vHist):
    xpos = []
    # endOfLastRun = c.x
    for i, _ in enumerate(vHist[:-1]):
        if not i and vHist[i]:
            xpos.append(c.x + i)

        # if vHist[i] and not vHist[i+1]:
        #     endOfLastRun = c.x + i

        elif not vHist[i] and vHist[i + 1]:
            xpos.append(c.x + i)

    xpos.append(c.x + c.width)
    divisions = []
    margin = 5
    for i, _ in enumerate(xpos[:-1]):
        l_pos, r_pos = max(xpos[i] - margin, c.x), xpos[i+1] - margin
        if(i == len(xpos) - 2):
            r_pos = xpos[i+1]
        newDivision = BaseComponent([l_pos, r_pos, c.y, c.y+c.height])
        divisions.append(newDivision)

    return divisions


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


def slice_image(image, boundingBoxes):
    slicedImage = []
    for box in boundingBoxes:
        [Xmin, Xmax, Ymin, Ymax] = box
        slicedImage.append(image[Ymin:Ymax, Xmin:Xmax])
    return slicedImage


def divide_beams(baseComponents, image, staffDim):
    allBaseComponents = []
    for cmp in baseComponents:
        slicedImage = image[cmp.y:cmp.y+cmp.height, cmp.x:cmp.x+cmp.width]
        heads = extract_heads(slicedImage, staffDim)
        vHist = get_vertical_projection(heads) > 0

        # show_images([slicedImage, heads])

        numHeads = get_number_of_heads(vHist)
        if numHeads > 1:
            components = divide_component(cmp, vHist)
            allBaseComponents.extend(components)
        else:
            allBaseComponents.append(cmp)

    # Sort components according to xpos
    allBaseComponents.sort(key=BaseComponent.sort_x_key)
    return allBaseComponents


def segment_image(image):
    lineImage, staffDim = extract_staff_lines(image)
    sanitized, mask, closed = sanitize_sheet(image)

    # Get base of components from boundingBoxes
    #ar_low, ar_high = staffDim[2] * 0.01, staffDim[2] * 0.105
    #boundingBoxes = get_bounding_boxes(closed, ar_low, ar_high)
    boundingBoxes = get_bounding_boxes(closed, 0.2)

    baseComponents = get_base_components(boundingBoxes)
    # Cut beams into notes

    baseComponents = divide_beams(baseComponents, sanitized, staffDim)

    # Retrieving image segments
    segments = []
    for cmp in baseComponents:
        segments.append(sanitized[cmp.slice])

    return np.array(segments), sanitized, staffDim, lineImage


def get_first_run(hist):
    '''
    @return A slice representing the run
    '''
    run = [-1, -1]
    for p, v in enumerate(hist):
        if run[0] < 0:
            if v:
                run[0] = p

        else:
            if not v:
                run[1] = p
                break

    return slice(run[0], run[1])


def get_base_components(boundingBoxes):
    '''
    This function also sorts base components on their x position
    '''
    baseComponents = []
    for box in boundingBoxes:
        component = BaseComponent(box)
        baseComponents.append(component)

    baseComponents.sort(key=BaseComponent.sort_x_key)
    return baseComponents


def sanitize_sheet(image):
    '''
    @return (Sanitized image, Image after closing)
    '''
    linesOnly, staffDim = extract_staff_lines(image)

    # Image - Lines
    removedLines = remove_staff_lines(image, linesOnly, staffDim)
    closedNotes = close_notes(removedLines, staffDim)

    # Clef removal
    vHist = get_vertical_projection(closedNotes)
    firstRun = get_first_run(vHist)
    closedNotes[:, firstRun] = 0

    # This step automatically removes barlines
    masked, mask = mask_image(closedNotes, removedLines)

    return masked, mask, closedNotes


def analyze_notes(noteImage, lineImage, staffDim):
    '''
    Log how far the bounding box is from the first line
    And whether it's over or under it
    '''
    below = ['f2', 'e2', 'd2', 'c2', 'b', 'a', 'g', 'f', 'e', 'd', 'c']
    above = ['f2', 'g2', 'a2', 'b2']

    staffSpacing = staffDim[2]
    firstLine = np.argmax(lineImage) // lineImage.shape[1]
    boundingBoxes = get_bounding_boxes(noteImage)
    boundingBoxes.sort()

    notes = []
    for _, _, yl, yh in boundingBoxes:
        mid = (yl + yh) // 2
        distance = abs(mid - firstLine)
        distance /= staffSpacing / 2
        distance = int(round(distance))

        if mid >= firstLine:
            notes.append(below[distance])

        else:
            notes.append(above[distance])

    return notes


def read_and_threshold_image(path):
    image = imread(path)
    if len(image.shape) > 2:
        image = (rgb2gray(image) * 255).astype(np.uint8)

    return image < threshold_otsu(image)


def split_bars(image):
    '''
    @return A list of images, each containing one bar
    '''
    lineImage, staffDim = extract_staff_lines(image)
    staffSpacing = staffDim[2]

    lineRows = np.unique(np.where(lineImage)[0])
    splits = [0]

    # Find run spacing larger than staff spacing
    for i, row in enumerate(lineRows[:-1]):
        spacing = lineRows[i+1] - row

        # Split at the midpoint
        if spacing > 8 * staffSpacing:
            splits.append(row + spacing//2)

    splits.append(image.shape[0])
    groups = []
    for i, sp in enumerate(splits[:-1]):
        groups.append(image[sp: splits[i+1]])

    return groups


# Read json manifest
# For each image
#   Segment image
#   For each segment
#       Map segment to json key
#       Create segment folder if not found
#       Append segment image to folder
def generate_dataset(inputDirectory, outputDirectory):
    '''
    inputDirectory should contain all images used for segmentation and dataset generation.
    It should also contain a `manifest.json` file which contains each image's details.

    The JSON file should be an array of objects with the format:
        path: <Relative path of the image file to inputDirectory>,
        segments: [
            <Symbol name. i.e.: 1_4, 1_8 >,
            <Symbol name. i.e.: 1_4, 1_8 >,
            <Symbol name. i.e.: 1_4, 1_8 >,
            <Symbol name. i.e.: 1_4, 1_8 >,
        ]
    '''
    jsonPath = os.path.join(inputDirectory, 'manifest.json')

    if os.path.exists(outputDirectory):
        rmtree(outputDirectory)

    os.makedirs(outputDirectory)

    with open(jsonPath, 'r') as jf:
        manifest = json.load(jf)

    counters = {}
    for image in manifest:
        path = os.path.join(inputDirectory, image['path'])
        data = read_and_threshold_image(path)

        segments, _, _, _ = segment_image(data)

        for record, segment in zip(image['segments'], segments):
            path = os.path.join(outputDirectory, record)
            if not os.path.exists(path):
                os.makedirs(path)

            if record not in counters:
                counters[record] = 0

            # copies = len(glob(os.path.join(path, '*')))
            # fullPath = os.path.join(path, str(copies))

            fullPath = os.path.join(
                path, f'{image["path"]}-{counters[record]}')
            counters[record] += 1

            imsave(f'{fullPath}.png', segment.astype(np.uint8) * 255)


def save_segments(segments):
    for i, seg in enumerate(segments):
        segment = seg.astype(np.uint8) * 255
        if not os.path.exists('samples'):
            os.makedirs('samples')
        imsave(f'samples/beam{i}.jpg', segment)

# Takes an image as an input
# Returns array of images separating staffs


def separate_multiple_staffs(image):
    binaryImage = image < np.mean(image)
    horizontal_histogram = get_horizontal_projection(binaryImage)
    threshVal = np.median(horizontal_histogram[horizontal_histogram[np.nonzero(
        horizontal_histogram)].argsort()[:9]])

    entered_peak = False
    begIndex = 0
    finishIndex = 0
    slicing_ranges = []

    for index, i in enumerate(horizontal_histogram):
        if i > threshVal and not entered_peak:
            entered_peak = True
            begIndex = index
        if i <= threshVal and entered_peak:
            finishIndex = index
            entered_peak = False
            slicing_ranges.append((begIndex, finishIndex))

    slicedImgs = []
    for pair in slicing_ranges:
        slicedImgs.append(image[pair[0]:pair[1], :])

    return slicedImgs
