from utility import *

image = (imread(sys.argv[1], as_gray=True) * 255).astype(np.uint8)
image = image < threshold_otsu(image)

sanitized, closed = sanitize_sheet(image)

show_images_columns(
    [image, sanitized],
    ['Original Image', 'Sanitized']
)

# Get base of components from boundingBoxes
boundingBoxes = get_bounding_boxes(closed)
baseComponents = get_base_components(boundingBoxes)

# Cut beams into notes
baseComponents = divide_beams(baseComponents, image)

# Sort components according to xpos
baseComponents.sort(key=BaseComponent.sort_x_key)

# Show all components
allSegments = []
for cmp in baseComponents:
    allSegments.append(sanitized[cmp.get_slice()])

show_images(allSegments)

save_segments(allSegments)
