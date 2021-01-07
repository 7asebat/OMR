import sys
import os
import json
from glob import glob
from shutil import rmtree
from skimage.morphology import binary_opening, binary_closing
import numpy as np

import Utility
import Processing
import Display
from Classifier import Classifier
from Component import Note


def demo_chord(chord, sanitized, lineImage, staffDim):
    chordImg, tones = Processing.process_chord(chord, sanitized, lineImage, staffDim)
    print(tones)
    processed = np.copy(sanitized)
    processed[chord.slice] = chordImg

    xl, xh, yl, yh = chord.box
    yl += chord.height // 4;

    chord = Note((xl, xh, yl, yh))
    Classifier.assign_flagged_note_timing(processed, chord)
    Display.show_images([sanitized[chord.slice], processed[chord.slice]])
    print(chord)


def demo_segmentation(inputPath):
    image = Utility.read_and_threshold_image(inputPath)

    groups = Processing.split_bars(image)
    Display.show_images(groups, [f'Group #{i}' for i in range(len(groups))])

    for i, group in enumerate(groups):
        components, sanitized, staffDim, lineImage = Processing.segment_image(
            group)

        Display.show_images_columns([group, sanitized],
                                    ['Original Image', 'Sanitized'],
                                    f'Group #{i}')

        # Showing note heads
        noteImage = Processing.extract_heads(sanitized, staffDim)
        Display.show_images_columns([sanitized, noteImage | lineImage],
                                    ['Sanitized Image', 'Note Heads on Staff Lines'],
                                    f'Group #{i}')

        Display.show_images([sanitized[cmp.slice] for cmp in components])


def demo_classification(inputPath):
    Classifier.load_classifiers({
        'meter_other': {
            'path': 'classifiers/classifier_meter_not_meter',
            'featureSet': 'hog'
        },
        'meter_time': {
            'path': 'classifiers/classifier_meter',
            'featureSet': 'hog'
        },
        'note_accidental': {
            'path': 'classifiers/classifier_notes_accidentals',
            'featureSet': 'hog'
        },
        'accidental_kind': {
            'path': 'classifiers/classifier_accidentals',
            'featureSet': 'hog'
        },
        'note_filled': {
            'path': 'classifiers/classifier_holes_old',
            'featureSet': 'hog'
        },
        'flagged_note_timing': {
            'path': 'classifiers/classifier_flags',
            'featureSet': 'hog'
        },
        'hollow_note_timing': {
            'path': 'classifiers/classifier_hollow',
            'featureSet': 'image_weight'
        },
        'beamed_note_timing': {
            'path': 'classifiers/classifier_beams',
            'featureSet': 'weighted_line_peaks'
            # 'path': 'classifiers/classifier_iterative_skeleton',
            # 'featureSet': 'iterative_skeleton'
        },
    })

    image = Utility.read_and_threshold_image(inputPath)
    groups = Processing.split_bars(image)

    print(inputPath, end=':\n\t')
    for group in groups:
        components, sanitized, staffDim, lineImage = Processing.segment_image(
            group)

        Classifier.assign_components(sanitized, components, staffDim)
        Processing.join_meters(components)
        Processing.bind_accidentals_to_following_notes(components)

        Processing.assign_note_tones(
            components, sanitized, lineImage, staffDim)
        print(Display.get_guido_notation(components), end='\n\t')

        # demo_chord(copmonents[1], sanitized, lineImage, staffDim)


    print('\n\n')

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
        data = Utility.read_and_threshold_image(path)

        components, sanitized, _, _ = Processing.segment_image(data)

        for record, component in zip(image['segments'], components):
            path = os.path.join(outputDirectory, record)
            if not os.path.exists(path):
                os.makedirs(path)

            if record not in counters:
                counters[record] = 0

            fullPath = os.path.join(
                path, f'{image["path"]}-{counters[record]}')
            counters[record] += 1

            segment = sanitized[component.slice]

            Utility.imsave(f'{fullPath}.png',
                           segment.astype(Utility.np.uint8) * 255)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        if sys.argv[1] == 'g':
            print('GENERATING...')
            generate_dataset(sys.argv[2], 'dataset')
        elif sys.argv[1] == 's':
            demo_segmentation(sys.argv[2])

    else:
        demo_classification(sys.argv[1])