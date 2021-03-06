__author__ = 'Daguerreo'

import numpy as np
import cv2
import os

class Util:
    def __init__(self):
        self.history = 50
        # n frame per diventare sfondo, n of gaussian, soglia backgroun-foreground
        self.fgbg = cv2.BackgroundSubtractorMOG(self.history,3,0.2)
        # self.fgbg = cv2.BackgroundSubtractorMOG2(5,0.2,True)

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

    def getMaskHSV(self, frame, background, channel='s'):
        if channel == 'h':
            c = 0
        elif channel == 's':
            c = 1
        elif channel == 'v':
            c = 2
        else:
            c = 0
            print 'warning: wrong channel on getMaskHSV'

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        background = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
        ch = frame[:,:,c]
        bg = background[:,:,c]

        mask = cv2.absdiff(ch,bg)
        mask = cv2.threshold(mask,30,255,cv2.THRESH_BINARY)[1]
        mask = cv2.medianBlur(mask, 7)

        element = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, element)
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, element)
        mask = cv2.medianBlur(mask, 9)

        return mask

    def getMaskMog(self, frame):
        fgmask = self.fgbg.apply(frame,learningRate=1.0/self.history)
        element = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, element)
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, element)
        fgmask = cv2.medianBlur(fgmask, 9)

        return fgmask

    def delFolderContent(self,path):
        for the_file in os.listdir(path):
            file_path = os.path.join(path, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                #elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception, e:
                print e
        return
