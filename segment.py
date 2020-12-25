from utility import *

image = (imread(sys.argv[1], as_gray=True) * 255).astype(np.uint8)
image = image < threshold_otsu(image)

linesOnly, staffDim = extract_staff_lines(image)

# Image - Lines
removedLines = remove_staff_lines(image, linesOnly, staffDim)
connectedNotes = connect_notes(removedLines, staffDim)

# Clef removal
firstRun = get_first_run(connectedNotes)
connectedNotes[:, firstRun] = 0

masked, mask = mask_image(connectedNotes, removedLines)

show_images_2(
    [image, removedLines, connectedNotes, mask],
    ['Original', 'Image - Lines', 'Closing(Image - Lines)', 'Segmentation']
)

show_images(
    [image, masked],
    ['Original Image', 'Masking']
)


# Get base of components from boundingBoxes
boundingBoxes = get_bounding_boxes(connectedNotes)
baseComponents = get_base_components(boundingBoxes)

# Cut beams into notes
baseComponents = divide_beams(baseComponents, image)

# Filter components according to AR to remove bar lines
#baseComponents = remove_vertical_bar_components(baseComponents)

# Sort components according to xpos
baseComponents.sort(key=BaseComponent.sort_x_key)

# Show all components
allSegments = []
for v in baseComponents:
    allSegments.append(masked[v.y:v.y+v.height, v.x:v.x+v.width])
show_images(allSegments)
