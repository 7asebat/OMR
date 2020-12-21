from utility import *

image = (imread(sys.argv[1], as_gray=True) * 255).astype(np.uint8)
image = image < threshold_otsu(image)

linesOnly, staffDim = extract_staff_lines(image)

# Image - Lines
removedLines = np.where(linesOnly, False, image)
connectedNotes = connect_notes(removedLines, staffDim)
boundingBoxes = get_bounding_boxes(connectedNotes)

show_images(
    [image, linesOnly, removedLines, connectedNotes],
    ['Original', 'Lines Only', 'Image - Lines', 'Closing(Image - Lines)']
)

dirtyMasked = mask_image(connectedNotes, image)
masked = remove_non_vertical_protrusions(dirtyMasked, staffDim)

show_images(
    [image, masked],
    ['Original Image', 'Masking']
)

# Get slices of masked
slicedBoxesofImage = slice_image(image, boundingBoxes)
slicedBoxesOfMasked = slice_image(masked, boundingBoxes)

# Segment everything
allSegments = segment_image(
    slicedBoxesofImage, slicedBoxesOfMasked, boundingBoxes)
show_images(allSegments)
