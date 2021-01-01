from utility import *


def main():
    image = (imread(sys.argv[1], as_gray=True) * 255).astype(np.uint8)
    image = image < threshold_otsu(image)

    staffGroups = split_staff_groups(image)
    show_images_columns(staffGroups, None, sys.argv[1])
    for i, group in enumerate(staffGroups):
        lineImage, staffDim = extract_staff_lines(group)
        sanitized, closed = sanitize_sheet(group)

        show_images_columns([group, sanitized],
                            ['Original Image', 'Sanitized'],
                            f'Group #{i}')

        # Get base of components from boundingBoxes
        boundingBoxes = get_bounding_boxes(closed)
        baseComponents = get_base_components(boundingBoxes)

        # Cut beams into notes
        baseComponents = divide_beams(baseComponents, group, staffDim)

        # Sort components according to xpos
        baseComponents.sort(key=BaseComponent.sort_x_key)

        # Showing note heads
        noteImage = extract_heads(sanitized, staffDim)
        print(analyze_notes(noteImage, lineImage, staffDim))
        show_images_columns([sanitized, noteImage | lineImage],
                            ['Sanitized Image', 'Note Heads on Staff Lines'],
                            f'Group #{i}')

        # Show all components
        segments = []
        for cmp in baseComponents:
            segments.append(sanitized[cmp.slice])
        show_images(segments)


if __name__ == "__main__":
    main()
