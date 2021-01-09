import os
import json
from glob import glob
from shutil import rmtree
import numpy as np
from skimage.morphology import binary_opening, binary_closing

import Preprocessing
import Utility
import Pipeline.Standard
import Pipeline.Augmented


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
    # Read json manifest
    # For each image
    #   Segment image
    #   For each segment
    #       Map segment to json key
    #       Create segment folder if not found
    #       Append segment image to folder
    jsonPath = os.path.join(inputDirectory, 'manifest.json')

    if os.path.exists(outputDirectory):
        rmtree(outputDirectory)

    os.makedirs(outputDirectory)

    with open(jsonPath, 'r') as jf:
        manifest = json.load(jf)

    counters = {}
    for image in manifest:
        path = os.path.join(inputDirectory, image['path'])
        data, useAugmented = Preprocessing.read_and_preprocess_image(path)
        Processing = Pipeline.Augmented if useAugmented else Pipeline.Standard

        components, sanitized, _, _, _ = Processing.segment_image(data)

        for record, component in zip(image['segments'], components):
            path = os.path.join(outputDirectory, record)
            if not os.path.exists(path):
                os.makedirs(path)

            if record not in counters:
                counters[record] = 0

            fullPath = os.path.join(path, f'{image["path"]}-{counters[record]}')
            counters[record] += 1

            segment = sanitized[component.slice]

            Utility.imsave(f'{fullPath}.png', segment.astype(np.uint8) * 255)
