from cv2 import copyMakeBorder, BORDER_CONSTANT, getStructuringElement, MORPH_ELLIPSE, morphologyEx, MORPH_OPEN, MORPH_ERODE, MORPH_CLOSE
from skimage.morphology import binary_closing, binary_dilation, binary_erosion, binary_opening, disk, selem
from scipy.signal import find_peaks
from scipy.ndimage.interpolation import rotate
import numpy as np

import Utility
from Classifier import Classifier
from Component import BaseComponent, Note, Meter, Accidental, Chord
from Display import show_images
from Segmentation import extract_heads, get_number_of_heads, detect_chord, detect_art_dots

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def extract_staff_lines(image):
    '''
    @return (LineImage, (width, length, spacing))
    '''
    # Staff lines using a horizontal histogram
    # Get staff line length
    # Threshold image based on said length

    if(image.max() == 0):
        return image, (0, 0, 0)

    linesOnly = np.copy(image)

    hHist = Utility.get_horizontal_projection(image)

    length = hHist.max()
    lengthThreshold = 0.85 * hHist.max()

    # Utility.visualize_histogram(hHist)
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

    # @note This step needs testing
    dotMask, _ = detect_art_dots(image, sanitized, staffDim)
    closed = np.where(dotMask, False, closed)

    # Get base of components from boundingBoxes
    boundingBoxes = Utility.get_bounding_boxes(closed, 0.2)

    # for box in boundingBoxes:
    #     area = (box[1] - box[0]) * (box[3] - box[2])
    #     if area <= 9:
    #         boundingBoxes.remove(box)

    baseComponents = Utility.get_base_components(boundingBoxes)

    # Cut beams into notes
    baseComponents = divide_beams(baseComponents, sanitized, staffDim)

    return baseComponents, sanitized, staffDim, lineImage


def sanitize_sheet(image):
    '''
    @return (Sanitized image, Closed image)
    '''
    imageAR = image.shape[1] / image.shape[0]
    segmentNum = int(np.ceil(imageAR * 5 + 0.5))
    segmentWidth = image.shape[1] // segmentNum
    processedImage = np.copy(image)
    closedImage = np.copy(image)

    if image.shape[1] % segmentNum:
        segmentNum += 1

    for i in range(segmentNum):
        slc_w = slice(i*segmentWidth, min((i+1) *
                                          segmentWidth, image.shape[1]))
        segment = image[:, slc_w]

        linesOnly, staffDim = extract_staff_lines(segment)

        k = np.zeros((3, 3), dtype='uint8')
        k[:, 3//2:3//2+1] = 1

        dilatedLinesOnly = binary_dilation(linesOnly, k)

        # Image - Lines
        removedLinesSeg = remove_staff_lines(
            segment, dilatedLinesOnly, staffDim)

        processedImage[:, slc_w] = removedLinesSeg
        closedNotesSeg = close_notes(removedLinesSeg, staffDim)
        closedImage[:, slc_w] = closedNotesSeg

    # Clef removal
    vHist = Utility.get_vertical_projection(processedImage)

    vHistThresh = vHist.max() * 0.60

    firstRun = Utility.get_first_run(vHist, vHistThresh)

    processedImage[:, firstRun] = 0
    closedImage[:, firstRun] = 0

    return processedImage, closedImage


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

        w_slc = slice(note.x-5, note.x+note.width+5)
        lineImage, staffDim = extract_staff_lines(originalImage[:, w_slc])

        staffSpacing = staffDim[2]
        staffThickness = staffDim[0]
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

        mid = get_vertical_center_of_gravity(head)
        # show_images([originalImage[:, w_slc]])

        try:
            mid += note.y
            note.tone = get_tone(mid, firstLine, staffSpacing, staffThickness)

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


def process_chord(chord, image, lineImage, staffDim):
    staffSpacing = staffDim[2]
    staffThickness = staffDim[0]
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

    headCenter = get_vertical_center_of_gravity(heads)

    furthest = 'below'
    if headCenter < chord.height // 2:
        furthest = 'above'

    boxes = Utility.get_bounding_boxes(heads)
    tones = []
    for _, _, yl, yh in boxes:
        mid = (yh + yl) // 2
        mid += chord.y
        tones.append(get_tone(mid, firstLine, staffSpacing, staffThickness))

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


def get_tone(mid, firstLine, staffSpacing, staffThickness):
    below = ['f2', 'e2', 'd2', 'c2', 'b1', 'a1', 'g1', 'f1', 'e1', 'd1', 'c1']
    above = ['f2', 'g2', 'a2', 'b2']

    below = ['f2', 'e2', 'd2', 'c2', 'b1', 'a1', 'g1', 'f1', 'e1', 'd1', 'c1']
    below_pos = []

    above = ['f2', 'g2', 'a2', 'b2']
    above_pos = []

    lineNow = firstLine + staffThickness / 2

    # print(mid, firstLine, staffSpacing, staffThickness)

    for _, _ in enumerate(below):
        below_pos.append(lineNow)
        lineNow += (staffSpacing / 2 + staffThickness / 2)

    lineNow = firstLine + staffSpacing / 2
    for _, _ in enumerate(above):
        above_pos.append(lineNow)
        lineNow -= (staffSpacing / 2 + staffThickness / 2)

    tone = ''
    if mid > firstLine:
        idx = (np.abs(below_pos - mid)).argmin()
        tone = below[idx]
    else:
        idx = (np.abs(above_pos - mid)).argmin()
        tone = above[idx]

    return tone


def get_vertical_center_of_gravity(image):
    # Calculate the image's vertical center of gravity
    avg = np.where(image)[0]
    return np.average(avg)
