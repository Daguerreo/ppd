__author__ = 'Daguerreo'

import cv2
import numpy as np
from skimage.feature import hog
from sklearn import svm
from sklearn.externals import joblib
import pickle

class Hog:
    def __init__(self, svm):
        self.svm = svm

    def setSVM(self, svm):
        self.svm = svm
        return

    def intersectionKernel(self,x, y):
        distances = np.zeros((x.shape[0], y.shape[0]))
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                distances[i, j] = np.sum(np.minimum(x[i, :], y[j, :]))
        return distances

    def checkIntersections(self, rect1, rect2, prob1, prob2, thresh_x,thresh_y):
        r1x1, r1y1, r1x2, r1y2 = rect1
        r2x1, r2y1, r2x2, r2y2 = rect2

        dx = np.abs(r1x1-r2x1)
        dy = np.abs(r1y1-r2y1)
        if dx < thresh_x or dy <thresh_y:
            if prob1 > prob2:
                return True, 1
                # cancella il secondo che gli hai buttato = rect2
            else:
                return True, 0
                # cancella il primo che gli hai buttato = rect1
        else:
            return False, None

    def draw_detections(self, img, rects, probaList, threshold, thickness=1):
        for i in range(len(rects)):
            x1 = rects[i][0]
            y1 = rects[i][1]
            x2 = rects[i][2]
            y2 = rects[i][3]
            if probaList[i] > threshold:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)

    def calcHog(self, window, orientation=9, pixelxcell=(8, 8), cellsxblock=(3, 3)):
        hist = hog(window, orientation, pixelxcell, cellsxblock)
        return hist

    def trainSVM(self, hogTrainingList, labels):
        self.svm.fit(hogTrainingList, labels)
        return

    # window e la finestrella dove potrebbe esserci una persona
    def run(self, window, label='None', orientation=9, pixelxcell=(8, 8), cellsxblock=(3, 3)):
        window=cv2.resize(window,(48,128))
        hist = self.calcHog(window, orientation, pixelxcell, cellsxblock)
        #lbl = self.svm.predict(hist)
        proba = self.svm.predict_proba(hist)
        if proba[0][1] > proba[0][0]:
            return True, proba
        else:
            return False, proba