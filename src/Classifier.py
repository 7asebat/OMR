import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier  # MLP is an NN
# from sklearn.svm import LinearSVC

from FeatureExtractor import FeatureExtractor
from Component import BaseComponent, Meter, Note, Accidental, Chord
import Processing
#from Processing import extract_heads, get_number_of_heads, detect_chord
from Utility import get_vertical_projection


class Classifier:
    __classifiers = {}

    def load_classifiers(classifiers):
        for c, cd in classifiers.items():
            Classifier.__classifiers[c] = Classifier(
                cd['path'], cd['featureSet'])

    def assign_components(image, baseComponents, staffDim):
        for i, cmp in enumerate(baseComponents):
            slc = image[cmp.slice]
            clf = Classifier.__classifiers['meter_other']
            tp = clf.extract_and_predict(slc)[0]

            # Meter
            if tp == 'meter':
                baseComponents[i] = Meter(cmp.box)
                Classifier.assign_meter_time(image, baseComponents[i])
                continue

            # Accidental or note

            clf = Classifier.__classifiers['note_accidental']
            tp = clf.extract_and_predict(slc)[0]

            if tp == 'Accidental':
                baseComponents[i] = Accidental(cmp.box)
                Classifier.assign_accidental_kind(image, baseComponents[i])
                continue

            # Beamed note
            if type(baseComponents[i]) is Note:
                Classifier.assign_beamed_note_timing(
                    image, baseComponents[i])
                continue

            # Flagged note or chord

            isChord = Processing.detect_chord(slc, staffDim)

            # Chord
            if isChord:
                chord = Chord(cmp.box)
                chord.filled = True
                baseComponents[i] = chord
                continue

            # Flagged note
            baseComponents[i] = Note(cmp.box)
            Classifier.assign_note_filled(
                image, baseComponents[i], staffDim)

    def assign_meter_time(image, meter):
        slc = image[meter.slice]
        clf = Classifier.__classifiers['meter_time']
        meter.time = clf.extract_and_predict(slc)[0]

    def assign_accidental_kind(image, accidental):
        slc = image[accidental.slice]
        clf = Classifier.__classifiers['accidental_kind']
        accidental.kind = clf.extract_and_predict(slc)[0]

    def assign_note_filled(image, note, staffDim):
        slc = image[note.slice]
        # clf = Classifier.__classifiers['note_filled']
        # note.filled = clf.extract_and_predict(slc)[0] == 'filled'

        heads = Processing.extract_heads_from_slice(slc, staffDim, filterAR=False)
        vHist = get_vertical_projection(heads) > 0
        numHeads = Processing.get_number_of_heads(vHist)
        if(numHeads > 0):
            note.filled = True

        if note.filled:
            # @todo Insert chorded note classification
            Classifier.assign_flagged_note_timing(image, note)

        else:
            Classifier.assign_hollow_note_timing(image, note)

    def assign_hollow_note_timing(image, note):
        slc = image[note.slice]
        clf = Classifier.__classifiers['hollow_note_timing']
        note.timing = clf.extract_and_predict(slc)[0]

    def assign_flagged_note_timing(image, note):
        slc = image[note.slice]
        clf = Classifier.__classifiers['flagged_note_timing']
        note.timing = clf.extract_and_predict(slc)[0]

    def assign_beamed_note_timing(image, note):
        slc = image[note.slice]
        clf = Classifier.__classifiers['beamed_note_timing']
        note.timing = clf.extract_and_predict(slc)[0]

    def __init__(self, path=None, featureSet=None):
        self.model = None
        self.featureSet = featureSet
        if path:
            self.model = pickle.load(open(path, 'rb'))

    def extract_and_predict(self, image, featureSet=None):
        # Use the classifier's configured feature set
        if not featureSet:
            featureSet = self.featureSet

        extractedFeatures = FeatureExtractor.extract(image, featureSet)
        return self.model.predict([extractedFeatures])

    def load_and_train(featureSet, datasetPath):
        raise NotImplementedError(
            'Model training has not yet been integrated.')
