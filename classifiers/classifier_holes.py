#!/usr/bin/env python
# coding: utf-8

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier  # MLP is an NN
from skimage.morphology import skeletonize
from sklearn import svm
from scipy.signal import *
import skimage.io as io
from matplotlib import pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import os
import random
import pickle

from sklearn.model_selection import train_test_split

path_to_dataset = r'dataset_all'
target_img_size = (32, 32)
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
labels_names = []


def extract_hog_features(img):
    img = cv2.resize(img, target_img_size)
    ret, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    win_size = (32, 32)
    cell_size = (4, 4)
    block_size_in_cells = (2, 2)

    block_size = (block_size_in_cells[1] * cell_size[1],
                  block_size_in_cells[0] * cell_size[0])
    block_stride = (cell_size[1], cell_size[0])
    nbins = 9  # Number of orientation bins
    hog = cv2.HOGDescriptor(win_size, block_size,
                            block_stride, cell_size, nbins)
    h = hog.compute(img)
    h = h.flatten()
    return h.flatten()


def extract_features(img, feature_set='hog'):
    if(feature_set == 'hog'):
        return extract_hog_features(img)


def load_dataset(feature_set='hog'):
    features = []
    labels = []
    labels_filenames = os.listdir(path_to_dataset)

    for i, ln in enumerate(labels_filenames):
        print("[INFO] label {}/{}".format(i+1, len(labels_filenames)))
        img_path = os.path.join(path_to_dataset, ln)
        img_filenames = os.listdir(img_path)
        for j, fn in enumerate(img_filenames):
            if fn.split('.')[-1] != 'png':
                continue
            if(ln == '1' or ln == '2'):
                labels.append('hole')
            else:
                labels.append('filled')

            path = os.path.join(img_path, fn)
            img = cv2.imread(path, 0)
            features.append(extract_features(img, feature_set))

            # show an update every 1,000 images
            if j > 0 and j % 1000 == 0:
                print("[INFO] image processed {}/{}".format(j, len(img_filenames)))
    return features, labels


def run_experiment(feature_set):
    model = svm.LinearSVC(random_state=random_seed, max_iter=10000, dual=True)
    model_name = "SVM"

    print('Loading dataset. This will take time...')
    features, labels = load_dataset(feature_set)

    print('Finished loading dataset..')
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=random_seed)

    print('############## Training', model_name, "##############")
    model.fit(train_features, train_labels)

    # Test the model on images it hasn't seen before
    accuracy = model.score(test_features, test_labels)
    print(model_name, 'accuracy:', accuracy*100, '%')

    pickle.dump(model, open('classifier_holes', 'wb'))
    print('############## Saved', model_name, "##############")


def load_predict(feature_set):
    model = pickle.load(open('classifier_holes', 'rb'))
    return model.predict(feature_set)


run_experiment('hog')
