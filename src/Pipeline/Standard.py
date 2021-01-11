from Pipeline.Common import *


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


def segment_image(image):
    lineImage, staffDim = extract_staff_lines(image)
    sanitized, closed = sanitize_sheet(image)

    dotMask, dotBoxes = detect_art_dots(image, sanitized, staffDim)
    closed = np.where(dotMask, False, closed)

    # Get base of components from boundingBoxes
    boundingBoxes = Utility.get_bounding_boxes(closed, 0.15)
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
    # @note Here we remove the minimum nonzero value of the histogram
    #       in case the entire histogram is lifted over 0
    vHist = Utility.get_vertical_projection(closedNotes)
    minimumProjection = np.min(vHist[np.nonzero(vHist)])
    vHist[vHist > 0] -= minimumProjection

    firstRun = Utility.get_first_run(vHist)
    while firstRun.stop - firstRun.start < 5:
        closedNotes[:, firstRun] = 0
        vHist[firstRun] = 0
        firstRun = Utility.get_first_run(vHist)

    closedNotes[:, firstRun] = 0
    vHist[firstRun] = 0

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

        mid += note.y
        note.tone = get_tone(mid, firstLine, staffSpacing)


def bind_accidentals_to_following_notes(components):
    for i, cmp in enumerate(components[:-1]):
        if type(cmp) is Accidental and type(components[i+1]) is Note:
            components[i+1].accidental = cmp.kind if cmp.kind != 'nat' else ''

    for cmp in components:
        if type(cmp) is Accidental:
            components.remove(cmp)


def process_chord(chord, image, lineImage, staffDim):
    staffSpacing = staffDim[2]
    firstLine = np.argmax(lineImage) // lineImage.shape[1]

    # Extract heads
    img = np.copy(image[chord.slice]).astype(np.uint8)
    heads = np.copy(img)

    # Use an elliptical structuring element
    w = staffSpacing // 2
    h = int(staffSpacing * 5.5/6) // 2
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
