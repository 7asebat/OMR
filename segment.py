from utility import *

def main():
    image = (imread(sys.argv[1], as_gray=True) * 255).astype(np.uint8)
    image = image < threshold_otsu(image)

    sanitized, closed, staffDim = sanitize_sheet(image)

    show_images_columns(
        [image, sanitized],
        ['Original Image', 'Sanitized'],
        sys.argv[1]
    )

    # Get base of components from boundingBoxes
    boundingBoxes = get_bounding_boxes(closed)
    baseComponents = get_base_components(boundingBoxes)

    # Cut beams into notes
    baseComponents = divide_beams(baseComponents, image, staffDim)

    # Sort components according to xpos
    baseComponents.sort(key=BaseComponent.sort_x_key)


    # Show all components
    segments = []
    for cmp in baseComponents:
        segments.append(sanitized[cmp.slice])
    show_images(segments)
    save_segments(segments)

    # Showing note heads
    show_images_rows([sanitized, extract_heads(sanitized, staffDim)])
    filteredHeads = []
    for s in segments:
        h = extract_heads(s, staffDim)
        filteredHeads.append(h)

    for s, h in zip(segments, filteredHeads):
        print(h)
        show_images_rows([s, h])

if __name__ == "__main__":
    main()