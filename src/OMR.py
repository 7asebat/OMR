import sys

import Utility
import Display
import Dataset
import Preprocessing
import Pipeline.Standard
import Pipeline.Augmented
from Classifier import Classifier


def demo_segmentation(inputPath):
    image, useAugmented = Preprocessing.read_and_preprocess_image(inputPath)
    Processing = Pipeline.Augmented if useAugmented else Pipeline.Standard

    groups = Processing.split_bars(image)
    Display.show_images(groups, [f'Group #{i}' for i in range(len(groups))])

    for i, group in enumerate(groups):
        components, sanitized, staffDim, lineImage, _ = Processing.segment_image(
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
    image, useAugmented = Preprocessing.read_and_preprocess_image(inputPath)
    Processing = Pipeline.Augmented if useAugmented else Pipeline.Standard

    Classifier.load_classifiers()

    groups = Processing.split_bars(image)

    output = []

    print(inputPath, end=':\n')
    for group in groups:
        components, sanitized, staffDim, lineImage, dotBoxes = Processing.segment_image(
            group)

        Classifier.assign_components(sanitized, components, staffDim)

        Processing.join_meters(components)
        Processing.bind_accidentals_to_following_notes(components)
        Processing.bind_dots_to_notes(components, dotBoxes)

        Processing.assign_note_tones(
            components, sanitized, lineImage, staffDim, group)
        output.append(Display.get_guido_notation(components))

    appendedString = ',\n\n'.join(output)
    finalOutput = f'{{\n{appendedString}\n}}'
    print(finalOutput)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        demo_classification(sys.argv[1])
    else:
        if sys.argv[1] == 'g':
            print('GENERATING...')
            Dataset.generate_dataset(sys.argv[2], 'dataset')
        elif sys.argv[1] == 's':
            demo_segmentation(sys.argv[2])
