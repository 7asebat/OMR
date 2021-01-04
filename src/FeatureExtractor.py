from skimage.morphology import skeletonize
from scipy.signal import *
import numpy as np
import cv2

target_img_size = (32, 32)

class FeatureExtractor:
    def __hog(image):
        image = image.astype(np.uint8) * 255
        image = cv2.resize(image, target_img_size)
        ret, image = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)
        win_size = (32, 32)
        cell_size = (4, 4)
        block_size_in_cells = (2, 2)

        block_size = (block_size_in_cells[1] * cell_size[1],
                    block_size_in_cells[0] * cell_size[0])
        block_stride = (cell_size[1], cell_size[0])
        nbins = 9  # Number of orientation bins
        hog = cv2.HOGDescriptor(win_size, block_size,
                                block_stride, cell_size, nbins)
        h = hog.compute(image)
        h = h.flatten()
        return h.flatten()


    def __weighted_line_peaks(image, expected=None):
        image = image.astype(np.uint8) * 255
        image = cv2.resize(image, target_img_size)
        ret, thresh = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)
        ks = 9
        kernel = np.zeros((ks, ks), dtype="uint8")
        kernel[ks//2: ks//2+1, :] = 1
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        ks = 3
        kernel = np.zeros((ks, ks), dtype="uint8")
        kernel[0:ks-1, ks//2: ks//2 + 1] = 1
        opened2 = cv2.morphologyEx(opened, cv2.MORPH_OPEN, kernel)

        skeleton = skeletonize(opened2)

        h_hist = np.sum(skeleton, 1)

        h_hist = h_hist.flatten()

        h_hist = np.concatenate(([min(h_hist)], h_hist, [min(h_hist)]))

        peaks = find_peaks(h_hist, distance=3)
        bigDist = 0
        smallDist = 0
        for i in range(len(peaks[0])):
            for j in range(i+1, len(peaks[0])):
                totalDist = abs(peaks[0][j] - peaks[0][i])
                if(totalDist > 15):
                    bigDist += 1
                else:
                    smallDist += 1

        return [len(peaks[0]) - 1, smallDist, bigDist]

    __features = {
        'hog': __hog,
        'weighted_line_peaks': __weighted_line_peaks
    }

    def extract(image, featureSet):
        if featureSet not in FeatureExtractor.__features:
            return None

        return FeatureExtractor.__features[featureSet](image)
