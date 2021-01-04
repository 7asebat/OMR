from FeatureExtractor import FeatureExtractor
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier  # MLP is an NN
from sklearn.svm import LinearSVC


class Classifier:
    def __init__(self, path):
        self.model = None
        if path:
            self.model = pickle.load(open(path, 'rb'))

    def load_and_train(featureSet, datasetPath):
        print('Loading dataset. This will take time...')
        # features, labels = load_dataset(path_to_dataset, feature_set)

        print('Finished loading dataset..')
        train_features, test_features, train_labels, test_labels = train_test_split(
            features, labels, test_size=0.2, random_state=random_seed)

    def extract_and_predict(self, image, featureSet='hog'):
        extractedFeatures = FeatureExtractor.extract(image, featureSet)
        return self.model.predict([extractedFeatures])
