import numpy as np
import sys

from skimage.draw import rectangle
from skimage.measure import find_contours
from skimage.io import imread, imsave
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray, rgba2rgb


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
        ar = (xValues.max() - xValues.min()) / \
            (yValues.max() - yValues.min())

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


def mask_image(connectedimage, image):
    '''
    Masks out all elements that aren't enclosed by a bounding box
    @note Extract the low ar value
    '''
    boundingBoxes = get_bounding_boxes(connectedimage, 0.2)
    mask = set_pixels(np.zeros(connectedimage.shape), boundingBoxes)
    return np.where(mask, image, False), mask


def keep_elements_in_ar_range(image, lower, upper):
    try:
        contours = find_contours(image, 0.8)
    except:
        return image

    filtered = np.copy(image)
    for c in contours:
        xValues = np.round(c[:, 1]).astype(int)
        yValues = np.round(c[:, 0]).astype(int)
        ar = (xValues.max() - xValues.min()) / \
            (yValues.max() - yValues.min())

        if ar < lower or ar > upper:
            rr, cc = rectangle(start=(yValues.min(), xValues.min()), end=(
                yValues.max(), xValues.max()), shape=filtered.shape)
            filtered[rr, cc] = 0

    return filtered


def read_and_threshold_image(path):
    image = imread(path)

    if len(image.shape) == 3:
        if image.shape[2] > 3:
            image = rgba2rgb(image)
        image = (rgb2gray(image) * 255).astype(np.uint8)

    return image < threshold_otsu(image)


def slice_image(image, boundingBoxes):
    slicedImage = []
    for box in boundingBoxes:
        [Xmin, Xmax, Ymin, Ymax] = box
        slicedImage.append(image[Ymin:Ymax, Xmin:Xmax])

    return slicedImage


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
