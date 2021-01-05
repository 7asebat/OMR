import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier  # MLP is an NN
from sklearn.svm import LinearSVC

from FeatureExtractor import FeatureExtractor
from Component import *


class Classifier:
    __classifiers = {}

    def load_classifiers(classifiers):
        for c, cd in classifiers.items():
            Classifier.__classifiers[c] = Classifier(cd['path'], cd['featureSet'])

    def assign_note_accidental(image, baseComponents):
        for i, cmp in enumerate(baseComponents):
            slc = image[cmp.slice]
            clf = Classifier.__classifiers['note_accidental']
            tp = clf.extract_and_predict(slc)[0]

            if tp == 'Accidental':
                baseComponents[i] = Accidental(cmp.box)
                Classifier.assign_accidental_kind(image, baseComponents[i])

            elif tp == 'Note':
                # Beamed note
                if type(baseComponents[i]) is Note:
                    Classifier.assign_beamed_note_timing(image, baseComponents[i])

                else:
                    baseComponents[i] = Note(cmp.box)
                    Classifier.assign_note_filled(image, baseComponents[i])

    def assign_accidental_kind(image, accidental):
        slc = image[accidental.slice]
        clf = Classifier.__classifiers['accidental_kind']
        accidental.kind = clf.extract_and_predict(slc)[0]

    def assign_note_filled(image, note):
        slc = image[note.slice]
        clf = Classifier.__classifiers['note_filled']
        note.filled = clf.extract_and_predict(slc)[0] == 'filled'

        if note.filled:
            Classifier.assign_flagged_note_timing(image, note)

        else:  # Hollow
            raise NotImplementedError('Hollow note timing classification has not yet been implemented.')

    def assign_flagged_note_timing(image, note):
        raise NotImplementedError('Flagged note timing classification has not yet been implemented.')

    def assign_beamed_note_timing(image, note):
        slc = image[note.slice]
        clf = Classifier.__classifiers['beamed_note_timing']
        note.timing = clf.extract_and_predict(slc, 'weighted_line_peaks')[0]

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
        pass
