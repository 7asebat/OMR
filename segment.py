from utility import *

image = (imread(sys.argv[1], as_gray=True) * 255).astype(np.uint8)
image = image < threshold_otsu(image)

linesOnly, staffDim = extract_staff_lines(image)

# Image - Lines
removedLines = remove_staff_lines(image, linesOnly, staffDim)
connectedNotes = connect_notes(removedLines, staffDim)
masked, mask = mask_image(connectedNotes, removedLines)

show_images_2(
    [image, removedLines, connectedNotes, mask],
    ['Original', 'Image - Lines', 'Closing(Image - Lines)', 'Segmentation']
)

show_images(
    [image, masked],
    ['Original Image', 'Masking']
)

# Clef removal
firstRun = get_first_run(connectedNotes)
connectedNotes[:, firstRun] = 0

# Get slices of masked
boundingBoxes = get_bounding_boxes(connectedNotes)
slicedImage = slice_image(image, boundingBoxes)
slicedMasked = slice_image(masked, boundingBoxes)

# Segment everything
allSegments = segment_image(slicedImage, slicedMasked)
show_images(allSegments)
