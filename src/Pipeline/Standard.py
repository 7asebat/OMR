import warnings
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

import sys
sys.path.append('..')


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

    # h_hist_img = np.zeros((len(hHist)+2, hHist.max()), dtype='uint8')
    # for i, val in enumerate(hHist):
    #     h_hist_img[i, :val] = True
    # show_images([h_hist_img])

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
    sanitized, closed = sanitize_sheet(image)

    dotMask, dotBoxes = detect_art_dots(image, sanitized, staffDim)
    closed = np.where(dotMask, False, closed)

    # Get base of components from boundingBoxes
    boundingBoxes = Utility.get_bounding_boxes(closed, 0.2)
    baseComponents = Utility.get_base_components(boundingBoxes)

    # Cut beams into notes
    baseComponents = divide_beams(baseComponents, sanitized, staffDim)

    return baseComponents, sanitized, staffDim, lineImage, dotBoxes


def sanitize_sheet(image):
    '''
    @return (Sanitized image, Closed image)
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
    masked, _ = Utility.mask_image(closedNotes, removedLines)

    return masked, closedNotes


def assign_note_tones(components, image, lineImage, staffDim, originalImage):
    '''
    Logs how far the note head's box is from the first line,
    and whether it's over or under it.
    '''
    for note in components:
        if type(note) is not Note and type(note) is not Chord:
            continue

        if type(note) is Chord:
            note = process_chord(note, image, lineImage, staffDim)
            continue

        staffSpacing = staffDim[2]
        firstLine = np.argmax(lineImage) // lineImage.shape[1]

        def extract_head(image):
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

        mid = Utility.get_vertical_center_of_gravity(head)

        try:
            mid += note.y
            note.tone = get_tone(mid, firstLine, staffSpacing)

        except Exception as e:
            print(e, end='\n\t')


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
        if spacing > 4 * staffSpacing:
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

    for cmp in components:
        if type(cmp) is Accidental:
            components.remove(cmp)


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


def process_chord(chord, image, lineImage, staffDim):
    staffSpacing = staffDim[2]
    firstLine = np.argmax(lineImage) // lineImage.shape[1]

    # Extract heads
    img = np.copy(image[chord.slice]).astype(np.uint8)
    heads = np.copy(img)

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

    headCenter = Utility.get_vertical_center_of_gravity(heads)

    furthest = 'below'
    if headCenter < chord.height // 2:
        furthest = 'above'

    boxes = Utility.get_bounding_boxes(heads)
    tones = []
    for _, _, yl, yh in boxes:
        mid = (yh + yl) // 2
        mid += chord.y
        tones.append(get_tone(mid, firstLine, staffSpacing))

    chord.tones = tones
    chord.tones.sort()

    # Get stem
    stem = np.copy(img)
    for xl, xh, yl, yh in boxes:
        slc = (slice(yl, yh), slice(xl, xh))
        stem[slc] = False
    stem = binary_opening(stem, selem=np.ones((3*staffSpacing, 1)))

    # Reset head image to original note head size
    w = staffSpacing
    h = int(staffSpacing * 5/6)
    SE_ellipse = getStructuringElement(MORPH_ELLIPSE, (w, h))
    SE_ellipse = rotate(SE_ellipse, angle=30)
    heads = morphologyEx(img, MORPH_OPEN, SE_ellipse,
                         borderType=BORDER_CONSTANT, borderValue=0)

    # Retain only the furthest head
    if furthest == 'below':
        furthest = max(boxes, key=lambda x: x[2])
    else:
        furthest = min(boxes, key=lambda x: x[2])
    oneHead = np.copy(heads)
    for box in boxes:
        if box != furthest:
            xl, xh, yl, yh = box
            slc = (slice(yl, yh), slice(xl, xh))
            oneHead[slc] = False

    oneHead = morphologyEx(oneHead, MORPH_OPEN, SE_ellipse,
                           borderType=BORDER_CONSTANT, borderValue=0)

    final = binary_closing(stem | oneHead)

    # crop xdim to fit the note precisely
    l, h = 0, 0
    vHist = np.sum(final, 0) > 0
    for i, _ in enumerate(vHist[:-1]):
        if(not vHist[i] and vHist[i+1] and l == 0):
            l = i
        if(vHist[i]):
            h = max(i, h)
    h = (chord.width - h)

    processed = np.copy(image)
    processed[chord.slice] = final

    xl, xh, yl, yh = chord.box

    note = Note((xl+l, xh-h, yl, yh))
    Classifier.assign_flagged_note_timing(processed, note)

    chord.timing = note.timing

    return chord


def get_tone(mid, firstLine, staffSpacing):
    below = ['f2', 'e2', 'd2', 'c2', 'b1', 'a1', 'g1', 'f1', 'e1', 'd1', 'c1']
    above = ['f2', 'g2', 'a2', 'b2']

    distance = abs(mid - firstLine)
    distance /= staffSpacing / 2
    distance = int(distance + 0.5)

    # @note This is a hacky fix which assumes that ONLY `c, b2` notes
    # are sometimes further than their standard distance
    bi = min(distance, len(below)-1)
    ai = min(distance, len(above)-1)
    return below[bi] if mid >= firstLine else above[ai]


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
