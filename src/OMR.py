import sys
import json
from glob import glob
from shutil import rmtree

import Utility
import Processing
import Display
from Classifier import Classifier


def demo_segmentation(inputPath):
    image = Utility.read_and_threshold_image(inputPath)

    groups = Processing.split_bars(image)
    # Display.show_images(groups)

    for i, group in enumerate(groups):
        segments, sanitized, staffDim, lineImage = Processing.segment_image(
            group)

        Display.show_images_columns([group, sanitized],
                                    ['Original Image', 'Sanitized'],
                                    f'Group #{i}')

        # Showing note heads
        noteImage = Processing.extract_heads(sanitized, staffDim)
        print(analyze_notes(noteImage, lineImage, staffDim))
        Display.show_images_columns([sanitized, noteImage | lineImage],
                                    ['Sanitized Image', 'Note Heads on Staff Lines'],
                                    f'Group #{i}')

        Display.show_images(segments)


def demo_classification(inputPath):
    image = Utility.read_and_threshold_image(inputPath)

    segments, _, _, _ = Processing.segment_image(image)

    classifier = Classifier('classifiers/classifier_notes_accidentals')

    for segment in segments[1:]:
        print(classifier.extract_and_predict(segment))

# Read json manifest
# For each image
#   Segment image
#   For each segment
#       Map segment to json key
#       Create segment folder if not found
#       Append segment image to folder
def generate_dataset(inputDirectory, outputDirectory):
    '''
    inputDirectory should contain all images used for segmentation and dataset generation.
    It should also contain a `manifest.json` file which contains each image's details.

    The JSON file should be an array of objects with the format:
        path: <Relative path of the image file to inputDirectory>,
        segments: [
            <Symbol name. i.e.: 1_4, 1_8 >,
            <Symbol name. i.e.: 1_4, 1_8 >,
            <Symbol name. i.e.: 1_4, 1_8 >,
            <Symbol name. i.e.: 1_4, 1_8 >,
        ]
    '''
    jsonPath = os.path.join(inputDirectory, 'manifest.json')

    if os.path.exists(outputDirectory):
        rmtree(outputDirectory)

    os.makedirs(outputDirectory)

    with open(jsonPath, 'r') as jf:
        manifest = json.load(jf)

    counters = {}
    for image in manifest:
        path = os.path.join(inputDirectory, image['path'])
        data = read_and_threshold_image(path)

        segments, _, _, _ = segment_image(data)

        for record, segment in zip(image['segments'], segments):
            path = os.path.join(outputDirectory, record)
            if not os.path.exists(path):
                os.makedirs(path)

            if record not in counters:
                counters[record] = 0

            # copies = len(glob(os.path.join(path, '*')))
            # fullPath = os.path.join(path, str(copies))

            fullPath = os.path.join(
                path, f'{image["path"]}-{counters[record]}')
            counters[record] += 1

            imsave(f'{fullPath}.png', segment.astype(np.uint8) * 255)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        print('GENERATING...')
        generate_dataset(sys.argv[2], 'dataset')

    else:
        demo_classification(sys.argv[1])
        # demo_segmentation(sys.argv[1])
