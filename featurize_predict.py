import numpy as np
import cv2
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score


def fit_classifier(X_train, Y_train):
    clf = RandomForestClassifier(
        n_estimators=1000, max_depth=None, min_samples_split=1)
    clf = clf.fit(X_train, Y_train)
    return clf


def evaluate_classifier(clf, X_test, Y_test):
    Y_preds = clf.predict(X_test)
    f1 = f1_score(Y_test, Y_preds)
    return f1


def featurize_PCA(X):
    pca = PCA(n_components=10)
    pca.fit(X)
    tX = pca.transform(X)
    return tX


def featurize_SIFT(X, num_kps=5):
    sift = cv2.xfeatures2d.SIFT_create()
    features = np.zeros(X.shape[0], 128 * num_kps)
    for i in range(X.shape[0]):
        kps, des = sift.detectAndCompute(X[i, :, :], None)
        for j in range(len(des) // 128):
            features[i, j * 128:(j + 1) * 128] = des[j * 128:(j + 1) * 128]


def featurize_HoG(X, tile_len=5, num_bins=10):
    features = np.zeros(X.shape[0],
                        (X.shape[1] // tile_len * num_bins) + (X.shape[2] // tile_len * num_bins))
    bin_size_degs = 180. / num_bins
    for i in X.shape[0]:
        im = X[0, :, :]
        Gx = ndimage.sobel(X, axis=1, mode='reflect', cval=0.)
        Gy = ndimage.sobel(X, axis=0, mode='reflect', cval=0.)
        row_offset = im.shape[0] // tile_len * num_bins
        for bi in range(im.shape[0] // tile_len):
            for bj in range(im.shape[1] // tile_len):
                hist_offset = (bi * row_offset) + (bj * num_bins)
                Gxs = np.ravel(Gx[bi:bi + tile_len, bj:bj + tile_len])
                Gys = np.ravel(Gy[bi:bi + tile_len, bj:bj + tile_len])
                angles = np.arctan(Gys / Gxs)
                mags = Gys + Gxs
                for angle, mag in zip(angles, mags):
                    bin_idx = angle // bin_size_degs
    features[i, hist_offset + bin_idx] += mag
                features[i, hist_offset:hist_offset + num_bins] *= 1. / np.linalg.norm(
                    features[i, hist_offset:hist_offset + num_bins])
    return features
