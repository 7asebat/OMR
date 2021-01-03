from utility import *


def demo_segmentation(inputPath):
    image = read_and_threshold_image(inputPath)

    groups = split_bars(image)
    show_images(groups)

    for i, group in enumerate(groups):
        segments, sanitized, staffDim, lineImage = segment_image(group)

        # Showing note heads
        #noteImage = extract_heads(sanitized, staffDim)
        # print(analyze_notes(noteImage, lineImage, staffDim))

        show_images_columns([group, sanitized],
                            ['Original Image', 'Sanitized'],
                            f'Group #{i}')

        # show_images_columns([sanitized, noteImage | lineImage],
        #                     ['Sanitized Image', 'Note Heads on Staff Lines'],
        #                     f'Group #{i}')

        show_images(segments)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        print('GENERATING...')
        generate_dataset(sys.argv[2], 'dataset')

    else:
        demo_segmentation(sys.argv[1])
