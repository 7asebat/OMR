from Display import show_images
from Component import BaseComponent, Note, Meter, Accidental, Chord
from Segmentation import extract_heads, get_number_of_heads, detect_chord, detect_art_dots
from Classifier import Classifier
import Utility
import numpy as np
from skimage.morphology import binary_closing, binary_dilation, binary_erosion, binary_opening, disk, selem
from scipy.signal import find_peaks
from scipy.ndimage.interpolation import rotate
from skimage.measure import find_contours
from cv2 import copyMakeBorder, BORDER_CONSTANT, getStructuringElement, MORPH_ELLIPSE, morphologyEx, MORPH_OPEN, MORPH_ERODE, MORPH_CLOSE


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


def close_notes(image, staffDim):
    '''
    Connects notes in an image with removed lines
    '''
    staffWidth = staffDim[0]
    SIZE = 5 * (staffWidth+1)
    SE_notes = np.ones((SIZE, 1))
    connectedNotes = binary_closing(image, SE_notes)  # Connect vertically
    return connectedNotes


def divide_component(c, vHist):
    xpos = []
    endOfLastRun = c.x
    for i, _ in enumerate(vHist[:-1]):
        if not i and vHist[i]:
            xpos.append(c.x + i)

        if vHist[i] and not vHist[i+1]:
            endOfLastRun = c.x + i

        elif not vHist[i] and vHist[i + 1]:
            # xpos.append(c.x + i)
            xpos.append((endOfLastRun + c.x + i) // 2)

    xpos.append(c.x + c.width)
    divisions = []
    for i, _ in enumerate(xpos[:-1]):
        newDivision = BaseComponent([xpos[i], xpos[i+1], c.y, c.y+c.height])
        divisions.append(newDivision)

    return divisions


def divide_component(c, vHist):
    xpos = []
    endOfLastRun = c.x
    for i, _ in enumerate(vHist[:-1]):
        if not i and vHist[i]:
            xpos.append(c.x + i)

        if vHist[i] and not vHist[i+1]:
            endOfLastRun = c.x + i

        elif not vHist[i] and vHist[i + 1]:
            # xpos.append(c.x + i)
            xpos.append((endOfLastRun + c.x + i) // 2)

    xpos.append(c.x + c.width)
    divisions = []
    for i, _ in enumerate(xpos[:-1]):
        newDivision = BaseComponent([xpos[i], xpos[i+1], c.y, c.y+c.height])
        divisions.append(newDivision)

    return divisions


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


def separate_multiple_staffs(image):
    '''
    Takes an image as an input
    Returns array of images separating staffs
    '''
    binaryImage = image < np.mean(image)
    horizontal_histogram = Utility.get_horizontal_projection(binaryImage)
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


def join_meters(baseComponents):
    meterList = [cmp for cmp in baseComponents if type(cmp) is Meter]
    if not meterList:
        return

    xl = min([m.x for m in meterList])
    xh = max([m.x+m.width for m in meterList])

    yl = min([m.y for m in meterList])
    yh = max([m.y+m.height for m in meterList])

    joinedMeter = Meter((xl, xh, yl, yh))
    joinedMeter.time = meterList[0].time

    baseComponents.insert(0, joinedMeter)
    for m in meterList:
        baseComponents.remove(m)


def bind_dots_to_notes(components, dotBoxes):
    notes = [note for note in components if type(note) is Note]

    for box in dotBoxes:
        def sq_distance(note):
            xl, xh, yl, yh = box
            noteMid = (note.x + note.width//2, note.y + note.height//2)
            boxMid = ((xl + xh) // 2, (yl + yh) // 2)
            diff = []
            for n, b in zip(noteMid, boxMid):
                diff.append((n-b)*(n-b))
            return sum(diff)

        closestNote = min(notes, key=sq_distance)
        closestNote.artdots += '.'


def remove_brace(image):
    contours = Utility.find_contours(image, 0.8)
    ar = []
    boundingBoxes = []

    for c in contours:
        xValues = np.round(c[:, 1]).astype(int)
        yValues = np.round(c[:, 0]).astype(int)
        box = (xValues.min(), xValues.max(), yValues.min(), yValues.max())
        dx = xValues.max() - xValues.min()
        dy = yValues.max() - yValues.min()

        if not dy:
            dy = 1
        if(dx != 0):
            ar.append(dx / dy)
        boundingBoxes.append(box)

    boxIndex = np.argmin(np.array(ar))

    boxHeight = boundingBoxes[boxIndex][-1] - boundingBoxes[boxIndex][-2]
    if(boxHeight >= 0.7*image.shape[0]):
        xl, xh, yl, yh = boundingBoxes[boxIndex]
        image[yl:yh, xl:xh] = False

    return image


def extract_staff_lines(image):
    '''
    @return (LineImage, (width, length, spacing))
    '''
    if not image.max():
        return image, (0, 0, 0)

    # Staff lines using a horizontal histogram
    # Get staff line length
    # Threshold image based on said length
    linesOnly = np.copy(image)

    hHist = Utility.get_horizontal_projection(image)
    length = hHist.max()
    lengthThreshold = 0.85 * hHist.max()

    hHist[hHist < lengthThreshold] = 0
    for r, val in enumerate(hHist):
        if not val:
            linesOnly[r] = 0

    # Get staff line width
    # @todo Replace this with median of runs
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

    runs = []
    startRun = 0
    for r, val in enumerate(hHist[:-1]):
        if hHist[r] == 0 and hHist[r+1] > 0:
            endRun = r+1 - startRun
            runs.append(endRun)
        elif hHist[r] > 0 and hHist[r+1] == 0:
            startRun = r

    spacing = int(np.median(runs))

    return linesOnly, (width, length, spacing)


def divide_beams(baseComponents, image, staffDim):
    allBaseComponents = []
    for cmp in baseComponents:
        slicedImage = image[cmp.y:cmp.y+cmp.height, cmp.x:cmp.x+cmp.width]
        heads = extract_heads(slicedImage, staffDim)
        vHist = Utility.get_vertical_projection(heads) > 0

        numHeads = get_number_of_heads(vHist)
        if numHeads > 1:
            components = divide_component(cmp, vHist)
            for i, cmp in enumerate(components):
                components[i] = Note(cmp.box)
                components[i].beamed = True
                components[i].filled = True

            allBaseComponents.extend(components)

        else:
            allBaseComponents.append(cmp)

    # Sort components according to xpos
    allBaseComponents.sort(key=BaseComponent.sort_x_key)
    return allBaseComponents
