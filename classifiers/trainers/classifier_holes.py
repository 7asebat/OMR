#!/usr/bin/env python
# coding: utf-8

from featureExtractor import *

path_to_dataset = r'dataset_holes_filled'


run_experiment('hog','classifier_holes',path_to_dataset)