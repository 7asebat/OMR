from Pipeline.Common import *


def segment_image(image):
    lineImage, staffDim = extract_staff_lines(image)

    sanitized, closed = sanitize_sheet(image)

    # @note This step needs testing
    dotMask, dotBoxes = detect_art_dots(image, sanitized, staffDim)
    closed = np.where(dotMask, False, closed)

    # Get base of components from boundingBoxes
    boundingBoxes = Utility.get_bounding_boxes(closed, 0.2)

    boundingBoxes2 = []

    for i, box in enumerate(boundingBoxes):
        area = (box[1] - box[0]) * (box[3] - box[2])
        if area > 500:
            boundingBoxes2.append(box)

    boundingBoxes = boundingBoxes2
    baseComponents = Utility.get_base_components(boundingBoxes)

    # Cut beams into notes
    baseComponents = divide_beams(baseComponents, sanitized, staffDim)

    return baseComponents, sanitized, staffDim, lineImage, dotBoxes


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

        mid = Utility.get_vertical_center_of_gravity(head)
        # show_images([originalImage[:, w_slc]])

        try:
            mid += note.y
            note.tone = get_tone(mid, firstLine, staffSpacing, staffThickness)

        except Exception as e:
            print(e, end='\n\t')


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

    headCenter = Utility.get_vertical_center_of_gravity(heads)

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
