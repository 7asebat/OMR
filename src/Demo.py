import sys
import Utility
import Display
import Dataset
import Preprocessing
import Pipeline.Standard
import Pipeline.Augmented
from Classifier import Classifier
from OMR import run_OMR


def demo_segmentation(inputPath):
    image, useAugmented = Preprocessing.read_and_preprocess_image(inputPath)
    Processing = Pipeline.Augmented if useAugmented else Pipeline.Standard

    image = Processing.remove_brace(image)

    groups = Processing.split_bars(image)
    Display.show_images(groups, [f'Group #{i}' for i in range(len(groups))])

    for i, group in enumerate(groups):
        components, sanitized, staffDim, lineImage, _ = Processing.segment_image(group)

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
    output = run_OMR(inputPath, './classifiers')
    print(inputPath, end=':\n')
    output = ',\n\n'.join(output)
    output = f'{{\n{output}\n}}'
    print(output)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        demo_classification(sys.argv[1])
    else:
        if sys.argv[1] == 'g':
            print('GENERATING...')
            Dataset.generate_dataset(sys.argv[2], 'dataset')
        elif sys.argv[1] == 's':
            demo_segmentation(sys.argv[2])
