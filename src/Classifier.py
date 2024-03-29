import pickle
from FeatureExtractor import FeatureExtractor
from Component import BaseComponent, Meter, Note, Accidental, Chord
from Segmentation import detect_chord, extract_heads, get_number_of_heads
from Utility import get_vertical_projection


class Classifier:
    __classifiers = {}

    def load_classifiers(classifiersPath):
        classifiers = {
            'meter_other': {
                'path': f'{classifiersPath}/classifier_meter_not_meter',
                'featureSet': 'hog'
            },
            'meter_time': {
                'path': f'{classifiersPath}/classifier_meter',
                'featureSet': 'hog'
            },
            'note_accidental': {
                'path': f'{classifiersPath}/classifier_notes_accidentals',
                'featureSet': 'hog'
            },
            'accidental_kind': {
                'path': f'{classifiersPath}/classifier_accidentals',
                'featureSet': 'hog'
            },
            'note_filled': {
                'path': f'{classifiersPath}/classifier_holes_old',
                'featureSet': 'hog'
            },
            'flagged_note_timing': {
                'path': f'{classifiersPath}/classifier_flags',
                'featureSet': 'hog'
            },
            'hollow_note_timing': {
                'path': f'{classifiersPath}/classifier_hollow',
                'featureSet': 'image_weight'
            },
            'beamed_note_timing': {
                'path': f'{classifiersPath}/classifier_beams',
                'featureSet': 'weighted_line_peaks'
            },
        }

        for c, cd in classifiers.items():
            Classifier.__classifiers[c] = Classifier(cd['path'], cd['featureSet'])

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
                Classifier.assign_beamed_note_timing(image, baseComponents[i])
                continue

            # Flagged note or chord
            isChord = detect_chord(slc, staffDim)

            # Chord
            if isChord:
                chord = Chord(cmp.box)
                chord.filled = True
                baseComponents[i] = chord
                continue

            # Flagged note
            baseComponents[i] = Note(cmp.box)
            Classifier.assign_note_filled(image, baseComponents[i], staffDim)

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

        heads = extract_heads(slc, staffDim, filterAR=False)
        vHist = get_vertical_projection(heads) > 0
        numHeads = get_number_of_heads(vHist)
        if(numHeads > 0):
            note.filled = True

        if note.filled:
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
        raise NotImplementedError('Model training has not yet been integrated.')
