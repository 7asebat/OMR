from utility import *

image = (imread(sys.argv[1], as_gray=True) * 255).astype(np.uint8)
image = image < threshold_otsu(image)

sanitized, closed, staffDim = sanitize_sheet(image)
heads = extract_heads(sanitized, staffDim)

show_images_columns(
    [image, sanitized, heads],
    ['Original Image', 'Sanitized', 'Heads only'],
    sys.argv[1]
)

sys.exit()

# Get base of components from boundingBoxes
boundingBoxes = get_bounding_boxes(closed)
baseComponents = get_base_components(boundingBoxes)

# Cut beams into notes
baseComponents = divide_beams(baseComponents, image, staffDim)

# Sort components according to xpos
baseComponents.sort(key=BaseComponent.sort_x_key)

# Show all components
allSegments = []
for cmp in baseComponents:
    allSegments.append(sanitized[cmp.slice])

show_images(allSegments)

save_segments(allSegments)
