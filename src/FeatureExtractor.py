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

    def __projection(image):
        image = image.astype(np.uint8) * 255
        image = cv2.resize(image, target_img_size)
        ret, thresh = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)

        h_hist = np.sum(thresh, 1)
        v_hist = np.sum(thresh, 0)

        h_hist = h_hist.flatten()
        v_hist = v_hist.flatten()

        return np.concatenate((h_hist, v_hist))

    def __weighted_line_peaks_2(image, expected=None):
        image = image.astype(np.uint8) * 255
        image = cv2.resize(image, target_img_size)
        ret, thresh = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)
        ks = 9
        kernel = np.zeros((ks, ks), dtype="uint8")
        kernel[ks//2: ks//2+1, :] = 1
        opened = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)

        ks = 3
        kernel = np.zeros((ks, ks), dtype="uint8")
        kernel[0:ks-1, ks//2: ks//2 + 1] = 1
        opened2 = cv2.morphologyEx(
            opened, cv2.MORPH_OPEN, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)

        skeleton = skeletonize(opened2)

        h_hist = np.sum(skeleton, 1)

        h_hist = h_hist.flatten()

        h_hist = np.concatenate(([min(h_hist)], h_hist, [min(h_hist)]))

        peaks = find_peaks(h_hist, prominence=3, distance=3)

        # calculate distance
        distances = [0] * 7
        divisor = 32 // (len(distances) - 1)
        for i in range(len(peaks[0])):
            for j in range(i+1, len(peaks[0])):
                x = abs(peaks[0][j] - peaks[0][i])
                bucket = min(x//divisor, len(distances) - 1)
                distances[bucket] += 1

        return [len(peaks[0]) - 1] + distances

    def __image_weight(image):
        image = image.astype(np.uint8) * 255
        image = cv2.resize(image, target_img_size)
        ret, thresh = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)

        white = np.argwhere(thresh).shape[0]

        ratio = white / (32 * 32)

        return [ratio]

    def __iterative_skeleton(image):
        def skeletonize_and_project(image):
            skeleton = skeletonize(image)
            projection = np.sum(skeleton, 1).flatten()
            projection = np.concatenate(
                ([min(projection)], projection, [min(projection)]))

            projectionImage = np.zeros(
                (image.shape[0]+2, image.shape[1]), dtype='uint8')
            for i, val in enumerate(projection):
                projectionImage[i, :val] = True

            return projectionImage, projection

        image = cv2.resize(image.astype(np.uint8), target_img_size)
        _, thresh = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)
        ks = 9
        kernel = np.zeros((ks, ks), dtype="uint8")
        kernel[ks//2: ks//2+1, :] = 1
        opened = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)

        ks = 3
        kernel = np.zeros((ks, ks), dtype="uint8")
        kernel[0:ks-1, ks//2: ks//2 + 1] = 1
        opened2 = cv2.morphologyEx(
            opened, cv2.MORPH_OPEN, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)

        it = opened2
        iterations = []
        for i in range(7):
            it, proj = skeletonize_and_project(it)
            iterations.append(it)

        # show_images(iterations)
        peaks = find_peaks(proj)

        # Number of flags - head
        return [len(peaks[0]) - 1]

    __features = {
        'hog': __hog,
        'weighted_line_peaks': __weighted_line_peaks,
        'projection': __projection,
        'weighted_line_peaks_2': __weighted_line_peaks_2,
        'image_weight': __image_weight,
        'projection': __projection,
        'iterative_skeleton': __iterative_skeleton,
    }

    def extract(image, featureSet):
        if featureSet not in FeatureExtractor.__features:
            return None

        return FeatureExtractor.__features[featureSet](image)
