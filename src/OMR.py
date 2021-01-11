from Classifier import Classifier
import Pipeline.Augmented
import Pipeline.Standard
import Preprocessing
import Dataset
import Display
import Utility
import sys
import warnings
warnings.filterwarnings('ignore')


def run_OMR(inputPath, classifiersPath):
    image, useAugmented = Preprocessing.read_and_preprocess_image(inputPath)
    Processing = Pipeline.Augmented if useAugmented else Pipeline.Standard

    Classifier.load_classifiers(classifiersPath)
    image = Processing.remove_brace(image)
    lineImage, staffDim = Processing.extract_staff_lines(image)
    groups = Processing.split_bars(image, lineImage, staffDim)

    output = []
    for group in groups:
        components, sanitized, staffDim, lineImage, dotBoxes = Processing.segment_image(group)

        Classifier.assign_components(sanitized, components, staffDim)

        Processing.join_meters(components)
        Processing.bind_accidentals_to_following_notes(components)
        Processing.bind_dots_to_notes(components, dotBoxes)

        Processing.assign_note_tones(components, sanitized, lineImage, staffDim, group)
        output.append(Display.get_guido_notation(components))

    return output
