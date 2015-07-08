__author__ = 'Daguerreo'

import cv2
import numpy as np
import DatasetManager
import Util
import Hog
import Logger
import os
from sklearn import svm

VERBOSE = True
REALTIME = True

# pathSequence = 'C:/Users/Daguerreo/Documents/dataset/Set_3/ID_76/Camera_1/Seq_1/'
pathSequence = 'sequences/Set_4/ID_123/Camera_8/Seq_1/'
pathFrame = 'video%04d.jpg'
pathComplete = pathSequence + pathFrame
pathTraining = "dataset/"
pathSave = "save/"
labelName = pathSave + "label.pickle"
trainListName = pathSave + "trainListLinear.pickle"
svmName = pathSave + "svmLinear.pickle"

util = Util.Util()
logger = Logger.Logger()
dm = DatasetManager.DatasetManager()
# mysvm = svm.SVC(kernel=intersectionKernel, C=10, probability=True)
mysvm = svm.SVC(probability=True)
myhog = Hog.Hog(mysvm)
framebuf = []
roimaskbuf = []
maskbuf = []
bgbuf = []

step = 15
orient = 8
matching_threshold = 0.9
mask_threshold = 0.3
minOverlap_x = 2*step
minOverlap_y = 5*step
scale = 1

def main():
    personsFound = 0
    index_frame = 1
    cap = cv2.VideoCapture(pathComplete)
    namesFrameList = dm.getFigNames(pathSequence)
    # fgbg = cv2.BackgroundSubtractorMOG()
    firstframe = cap.read()[1]
    firstframe = cv2.resize(firstframe,(0,0),None,0.8,0.8)
    average = np.float32(firstframe)
    cv2.accumulateWeighted(firstframe, average, 0.1)
    logger.timerStart()
    # train(mysvm, pathTraining,False,True,False,True,False,True)
    train(mysvm, pathTraining)

    while cap.isOpened():
        success, framergb = cap.read()
        if not success:
            break

        framergb = cv2.resize(framergb,(0,0),None,0.8,0.8)

        # fgmask = fgbg.apply(framergb)
        # element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, element)
        # element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
        # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, element)
        # fgmask = cv2.medianBlur(fgmask, 9)

        frame = cv2.cvtColor(framergb, cv2.COLOR_BGR2GRAY)
        cv2.accumulateWeighted(framergb, average, 0.02)
        background = cv2.convertScaleAbs(average)
        mask = util.getMaskHSV(framergb,background)
        # background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        # mask = util.getMask(frame, background)

        if REALTIME is False:
            maskbuf.append(mask)

        frame = cv2.blur(frame,(5,5))
        totalRect, p = calcHog(framergb, frame, mask, namesFrameList[index_frame])
        personsFound += p
        index_frame += 1

        if REALTIME is True:
            cv2.imshow('original', framergb)
            cv2.imshow('mask', mask)
            # cv2.imshow('mask2', fgmask)
            # cv2.imshow('h',h)
            c = cv2.waitKey(1)
            if c == ord(' '):
                break

    print 'Total Time: ' + str(logger.totalTime())
    print 'Rect Founds: ' + str(personsFound)

    if REALTIME is False:
        for i in range(0,len(framebuf)):
            cv2.imshow('original', framebuf[i])
            cv2.imshow('mask', maskbuf[i])
            c = cv2.waitKey(50)
            if c == ord(' '):
                break
    return

def train(svm, trainingPath, loadlbl=True, savelbl=False, loadtrain=True, savetrain=False, loadsvm=True, savesm=False):
    pix_x_cell = (16, 16)
    cell_x_block = (1, 1)
    hogTrainingList = []
    labelTrainingList=[]

    if loadlbl is True:
        trainingSet = logger.load(labelName)
    else:
        if VERBOSE is True:
            print 'start labeling'

        trainingSet = dm.sortDataset(trainingPath, 0, False)[0]

        if VERBOSE is True:
            print 'labeling complete in ' + str(logger.timerRound()) + ' ms'

    if savelbl is True:
        logger.save(trainingSet,labelName)

    if loadtrain is False:
        if VERBOSE is True:
            print 'start training'

        for i in range(len(trainingSet)):
            temp_img=cv2.imread(trainingSet[i].path, cv2.IMREAD_GRAYSCALE)
            hogTrainingList.append(myhog.calcHog(temp_img,orient,pix_x_cell,cell_x_block))
            labelTrainingList.append(trainingSet[i].label)

        if VERBOSE is True:
            print 'training complete in ' + str(logger.timerRound()) + ' ms'

        if savetrain is True:
            logger.save(labelTrainingList,trainListName)
    else:
        labelTrainingList = logger.load(trainListName)

    if loadsvm is False:
        myhog.trainSVM(hogTrainingList,labelTrainingList)

        if VERBOSE is True:
            print 'SVM training complete in ' + str(logger.timerRound()) + ' ms'

        if savesm is True:
            logger.save(svm,svmName)
    else:
        svm = logger.load(svmName)
        myhog.setSVM(svm)

    return

def calcHog(framergb, frame, mask, nameFrame):
    positionList=[]
    probaList=[]
    totalRect = 0
    widthROI = int(48*scale)
    heightROI = int(128*scale)
    pix_x_cell = (16, 16)
    cell_x_block = (1, 1)

    for i in range(0,frame.shape[0]-heightROI,step):
        for j in range(0,frame.shape[1]-widthROI,step):
             if (j+widthROI < frame.shape[1]) and (i+heightROI < frame.shape[0]):
                roi = mask[i:i+heightROI,j:j+widthROI]
                roimaskbuf.append(roi)
                if util.isThereMovement(roi,mask_threshold):
                    window = frame[i:i+heightROI,j:j+widthROI]
                    isPerson, proba = myhog.run(window,'Person',orient,pix_x_cell,cell_x_block)
                    totalRect += 1
                else:
                    continue

                if isPerson is True and proba[0][1] > matching_threshold:
                    positionList.append((j,i,j+widthROI,i+heightROI))
                    probaList.append(proba[0][1])
                    # if VERBOSE is True:
                    #     print 'person found: ' + str(proba)

    subjectLength=len(positionList)
    fuoriList=[]

    for i in range(subjectLength - 1, -1, -1):
        for j in range(subjectLength - 1, -1, -1):
            if i != j:
                check,k=myhog.checkIntersections(positionList[i],positionList[j],probaList[i],probaList[j],minOverlap_x,minOverlap_y)
                if check is True:
                    if k==0:
                        fuoriList.append(i)
                    else:
                        fuoriList.append(j)

    definitiveList=[]
    defproba=[]
    for i in range(0,len(positionList)):
        if i not in fuoriList:
            definitiveList.append(positionList[i])
            defproba.append(probaList[i])

    myhog.draw_detections(framergb, definitiveList, defproba, matching_threshold, 2)

    file = open('rects/'+os.path.basename(nameFrame)+'.gt','w')
    for f in definitiveList:
       file.write(str(f[0]) + ' ' + str(f[1]) + ' ' + str(f[2]) + ' ' + str(f[3]) + '\n')
    file.close()

    if REALTIME is False:
        framebuf.append(framergb)

    return totalRect, len(definitiveList)
########################################################################################
# program start here
main()
