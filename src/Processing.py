import sys
import numpy as np
from skimage.morphology import binary_closing, binary_dilation, binary_erosion, binary_opening, disk, selem
from scipy.signal import find_peaks
from scipy.ndimage.interpolation import rotate
import Utility
from Component import BaseComponent, Note, Meter, Accidental
from cv2 import copyMakeBorder, BORDER_CONSTANT, getStructuringElement, MORPH_ELLIPSE, morphologyEx, MORPH_OPEN

from Display import show_images

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def extract_staff_lines(image):
    '''
    @return (LineImage, (width, length, spacing))
    '''
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

    # Get staff line spacing
    # Find the space between any two consecutive runs
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
    staffSpacing = staffDim[2]
    vHist = Utility.get_vertical_projection(image) > 0
    numHeads = get_number_of_heads(vHist)

    closedImage = binary_closing(image)
    if numHeads > 1:
        closedImage = binary_closing(
            image, np.ones((10, 10), dtype='bool'))

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

    filteredSolidHeads = Utility.keep_elements_in_ar_range(
        solidHeads, 0.9, 1.5)

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
    # margin = 5
    for i, _ in enumerate(xpos[:-1]):
        # l_pos, r_pos = max(xpos[i] - margin, c.x), xpos[i+1] - margin
        # if(i == len(xpos) - 2):
        #     r_pos = xpos[i+1]
        # newDivision = BaseComponent([l_pos, r_pos, c.y, c.y+c.height])
        newDivision = BaseComponent([xpos[i], xpos[i+1], c.y, c.y+c.height])
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


def segment_image(image):
    lineImage, staffDim = extract_staff_lines(image)
    sanitized, _, closed = sanitize_sheet(image)

    # Get base of components from boundingBoxes
    boundingBoxes = Utility.get_bounding_boxes(closed, 0.2)
    baseComponents = Utility.get_base_components(boundingBoxes)

    # Cut beams into notes
    baseComponents = divide_beams(baseComponents, sanitized, staffDim)

    return baseComponents, sanitized, staffDim, lineImage


def sanitize_sheet(image):
    '''
    @return (Sanitized image, Sanitization mask, Image after closing)
    '''
    linesOnly, staffDim = extract_staff_lines(image)

    # Image - Lines
    removedLines = remove_staff_lines(image, linesOnly, staffDim)
    closedNotes = close_notes(removedLines, staffDim)

    # Clef removal
    vHist = Utility.get_vertical_projection(closedNotes)
    firstRun = Utility.get_first_run(vHist)
    closedNotes[:, firstRun] = 0

    # This step automatically removes barlines
    masked, mask = Utility.mask_image(closedNotes, removedLines)

    return masked, mask, closedNotes


def assign_note_tones(components, image, lineImage, staffDim):
    '''
    Logs how far the note head's box is from the first line,
    and whether it's over or under it.

    Raises `ValueError` if the supplied component is not a note.
    '''
    for note in components:
        if type(note) is not Note:
            continue

        below = ['f2', 'e2', 'd2', 'c2', 'b', 'a', 'g', 'f', 'e', 'd', 'c']
        above = ['f2', 'g2', 'a2', 'b2']
        staffSpacing = staffDim[2]
        firstLine = np.argmax(lineImage) // lineImage.shape[1]

        def extract_head(image):
            staffSpacing = staffDim[2]
            closedImage = image
            closedImage = binary_closing(image).astype(np.uint8)

            # Use an elliptical structuring element
            w = staffSpacing
            h = int(staffSpacing * 6/7)
            SE_ellipse = getStructuringElement(MORPH_ELLIPSE, (w, h))
            SE_ellipse = rotate(SE_ellipse, angle=30)

            # @note skimage sucks
            head = morphologyEx(closedImage, MORPH_OPEN, SE_ellipse,
                                borderType=BORDER_CONSTANT, borderValue=0)
            # show_images([closedImage, head])
            return head

        head = np.copy(image[note.slice])
        # @note False classification of filled notes results in
        # unneccessary closing
        if not note.filled:
            SIZE = (staffSpacing-3)//2
            SE_disk = disk(SIZE)
            head = binary_closing(head, SE_disk)
            head = binary_opening(head, SE_disk)

        head = extract_head(head)

        def get_mid(head):
            # Calculate the image's center of gravity
            avg = np.where(head)[0]
            return np.average(avg)

        mid = get_mid(head)

        try:
            mid += note.y
            distance = abs(mid - firstLine)
            distance /= staffSpacing / 2
            distance = int(distance + 0.5)

            # @note This is a hacky fix which assumes that ONLY `c, b2` notes
            # are sometimes further than their standard distance
            bi = min(distance, len(below)-1)
            ai = min(distance, len(above)-1)
            note.tone = below[bi] if mid >= firstLine else above[ai]

        except Exception as e:
            print(e, end='\n\t')


def analyze_notes(noteImage, lineImage, staffDim):
    '''
    Log how far the bounding box is from the first line
    And whether it's over or under it
    '''
    below = ['f2', 'e2', 'd2', 'c2', 'b', 'a', 'g', 'f', 'e', 'd', 'c']
    above = ['f2', 'g2', 'a2', 'b2']

    staffSpacing = staffDim[2]
    firstLine = np.argmax(lineImage) // lineImage.shape[1]
    boundingBoxes = Utility.get_bounding_boxes(noteImage)
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


def bind_accidentals_to_following_notes(components):
    for i, cmp in enumerate(components[:-1]):
        if type(cmp) is Accidental and type(components[i+1]) is Note:
            components[i+1].accidental = cmp.kind if cmp.kind != 'nat' else ''
