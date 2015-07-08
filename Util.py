__author__ = 'Daguerreo'

import numpy as np
import cv2

class Util:
    def __init__(self):
        pass

    def getROI(self, img, edge, sizex, sizey):
        x1 = edge[0]
        x2 = edge[0] + sizex
        y1 = edge[1]
        y2 = edge[1] + sizey

        return img[y1:y2, x1:x2]

    def isThereMovement(self, mask, threshold):
        arr = np.asarray(mask,float)
        size = arr.size
        sum = np.sum(arr)
        sum = sum.astype(float)/255.0
        ratio = sum/size

        if ratio > threshold:
            return True

        return False

    def getMask(self, frame, background):
        mask = cv2.absdiff(frame,background)
        mask = cv2.threshold(mask,40,255,cv2.THRESH_BINARY)[1]
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, element)
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, element)
        mask = cv2.medianBlur(mask, 9)
        return mask

    def getMaskHSV(self, frame, background):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        background = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
        frameh = frame[:,:,1]
        framev = frame[:,:,2]
        bgh = background[:,:,1]
        bgv = background[:,:,2]

        maskh = cv2.absdiff(frameh,bgh)
        maskh = cv2.threshold(maskh,30,255,cv2.THRESH_BINARY)[1]
        maskh = cv2.medianBlur(maskh, 5)
        maskv = cv2.absdiff(framev,bgv)
        maskv = cv2.threshold(maskv,30,255,cv2.THRESH_BINARY)[1]
        maskv = cv2.medianBlur(maskv, 5)
        return maskh, maskv
