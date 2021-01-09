import numpy as np
import os

from skimage.draw import rectangle
from skimage.measure import find_contours
from skimage.io import imread, imsave
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray, rgba2rgb
from Display import show_images

from Component import BaseComponent

# When provided with the correct format of the list of bounding_boxes, this section will set all pixels inside boxes in image_with_boxes


def set_pixels(image, bounding_boxes):
    image_with_boxes = np.copy(image)
    for box in bounding_boxes:
        Xmin, Xmax, Ymin, Ymax = box
        rr, cc = rectangle(start=(Ymin, Xmin), end=(
            Ymax, Xmax), shape=image.shape)
        image_with_boxes[rr, cc] = 1  # set color white
    return image_with_boxes


def get_bounding_boxes(image, lower=0, upper=np.inf, filterSubsets=True):
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
        dx = xValues.max() - xValues.min()
        dy = yValues.max() - yValues.min()

        if not dy:
            dy = 1
        ar = dx / dy

        # Filter out barlines
        # Keep larger bounding box
        # Filter overlapping bounding boxes
        if filterSubsets:
            insertNew = True
            if lower <= ar <= upper:
                for x in boundingBoxes:
                    if is_subset(x, box):
                        boundingBoxes.remove(x)
                    elif is_subset(box, x):
                        insertNew = False

                if insertNew:
                    boundingBoxes.append(box)
        else:
            if lower <= ar <= upper:
                boundingBoxes.append(box)

    return boundingBoxes


def get_vertical_projection(image):
    return np.array([image[:, col].sum() for col in range(image.shape[1])])


def get_vertical_projection_image(image):
    hist = get_vertical_projection(image)

    histimage = np.zeros((hist.max(), hist.shape[0]), dtype='uint8')
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
    boundingBoxes = get_bounding_boxes(connectedimage, 0.15)
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


def slice_image(image, boundingBoxes):
    slicedImage = []
    for box in boundingBoxes:
        [Xmin, Xmax, Ymin, Ymax] = box
        slicedImage.append(image[Ymin:Ymax, Xmin:Xmax])

    return slicedImage


def get_first_run(hist, thresh=0):
    '''
    @return A slice representing the run
    '''
    run = [-1, -1]
    foundPeak = False
    for p, v in enumerate(hist):
        if v > thresh:
            foundPeak = True

        if run[0] < 0:
            if v:
                run[0] = p

        else:
            if not v and foundPeak:
                run[1] = p
                break

    return slice(run[0], run[1])


def get_base_components(boundingBoxes):
    '''
    This function also sorts base components on their x position
    '''
    baseComponents = []
    for box in boundingBoxes:
        if box[1] - box[0] < 3 or box[3] - box[2] < 3:
            continue
        component = BaseComponent(box)
        baseComponents.append(component)

    baseComponents.sort(key=BaseComponent.sort_x_key)
    return baseComponents


def save_segments(segments):
    for i, seg in enumerate(segments):
        segment = seg.astype(np.uint8) * 255
        if not os.path.exists('samples'):
            os.makedirs('samples')
        imsave(f'samples/beam{i}.jpg', segment)


def get_vertical_center_of_gravity(image):
    # Calculate the image's vertical center of gravity
    avg = np.where(image)[0]
    return np.average(avg)
