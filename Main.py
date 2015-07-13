__author__ = 'Daguerreo'

import cv2
import numpy as np
import DatasetManager
import Util
import Hog
import Logger
import os
import Statistic
import datetime
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

VERBOSE = True
REALTIME = True
BATCH = False

pathFrame = 'video%04d.jpg'
pathTraining = "dataset/"
pathSave = "save/"
sequence = "sequences/"
groundtruth = "groundtruth/"
pathTest = "rects/"

pathFolder = "Set_3/ID_76/Camera_1/Seq_1/"
# pathFolder = "Set_4/ID_138/Camera_8/Seq_1/"
# pathFolder = "Set_4/ID_139/Camera_8/Seq_1/"

pathGT = groundtruth + pathFolder

labelName = pathSave + "label.pickle"
trainListName = pathSave + "trainList.pickle"
clfName = pathSave + "svm.pickle"

util = Util.Util()
logger = Logger.Logger()
dm = DatasetManager.DatasetManager()
stat=Statistic.Statistic()

# mysvm = svm.SVC(kernel=intersectionKernel, C=10, probability=True)
mysvm = svm.SVC(probability=True)
# mysvm = RandomForestClassifier(n_estimators=100)
myhog = Hog.Hog(mysvm)
framebuf = []
roimaskbuf = []
maskbuf = []
bgbuf = []

# cancella il contenuto di rects prima di iniziare
util.delFolderContent(pathTest)

step = 12
orient = 8
matching_threshold = 0.9
mask_threshold = 0.3
overlap_threshold = 0.5
edge_distance_threshold = 24
minOverlap_x = 2*step
minOverlap_y = 5*step
scale = 1.0
frameScale = 0.7

def main(pathfolder):
    pathSequence = sequence + pathfolder
    pathComplete = pathSequence + pathFrame
    personsFound = 0
    index_frame = 1
    cap = cv2.VideoCapture(pathComplete)
    namesFrameList = dm.getFigNames(pathSequence)
    firstframe = cap.read()[1]
    firstframe = cv2.resize(firstframe,(0,0),None,frameScale,frameScale)
    average = np.float32(firstframe)
    cv2.accumulateWeighted(firstframe, average, 0.2)
    logger.timerStart()
    # train(mysvm, pathTraining,True,False,False,True,False,True)
    train(mysvm, pathTraining)

    while cap.isOpened():
        success, framergb = cap.read()
        if not success:
            break

        framergb = cv2.resize(framergb,(0,0),None,frameScale,frameScale)

        frame = cv2.cvtColor(framergb, cv2.COLOR_BGR2GRAY)
        cv2.accumulateWeighted(framergb, average, 0.02)
        background = cv2.convertScaleAbs(average)
        mask = util.getMaskHSV(framergb,background)
        background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        maskgray = util.getMask(frame, background)
        # maskMog = util.getMaskMog(framergb)
        mask += maskgray

        if REALTIME is False and BATCH is False:
            maskbuf.append(mask)

        frame = cv2.blur(frame,(5,5))
        totalRect, p = calcHog(framergb, frame, mask, namesFrameList[index_frame])
        personsFound += p
        index_frame += 1

        if REALTIME is True and BATCH is False:
            cv2.imshow('original', framergb)
            cv2.imshow('mask', mask)
            # cv2.imshow('mog', maskMog)

            c = cv2.waitKey(1)
            if c == ord(' '):
                break
    time = logger.totalTime()
    print 'Total Time: ' + str(time)
    print 'Rect Founds: ' + str(personsFound)

    if REALTIME is False and BATCH is False:
        for i in range(0,len(framebuf)):
            cv2.imshow('original', framebuf[i])
            cv2.imshow('mask', maskbuf[i])
            c = cv2.waitKey(50)
            if c == ord(' '):
                break


    # TP, TN, FP, FN = stat.calcPositiveNegative(pathGT,pathTest,frameScale,24)
    # calcola la distanza dagli spigoli
    TP, TN, FP, FN = stat.calcPositiveNegative(pathGT,pathTest,frameScale,edge_distance_threshold)
    # calcola l'area di overlap tra i parametri
    # TP, TN, FP, FN = stat.calcStatisticsValues(pathGT,pathTest,frameScale,overlap_threshold)
    print datetime.datetime.now()
    print "TP = "+str(TP)+", TN = "+str(TN)+", FP = "+str(FP)+", FN = "+str(FN)
    accuracy,recall,precision,f1score=stat.calcPerformance(TP,TN,FP,FN)
    print "Accuracy = "+str(accuracy)
    print "Recall = "+str(recall)
    print "Precision = "+str(precision)
    print "F1 Score = "+str(f1score)

    return accuracy, recall, precision, f1score, time

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
        myhog.trainCLF(hogTrainingList,labelTrainingList)

        if VERBOSE is True:
            print 'SVM training complete in ' + str(logger.timerRound()) + ' ms'

        if savesm is True:
            logger.save(svm,clfName)
    else:
        svm = logger.load(clfName)
        myhog.setSVM(svm)

    return

def calcHog(framergb, frame, mask, nameFrame):
    positionList = []
    probaList = []
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

    subjectLength = len(positionList)
    fuoriList = []

    for i in range(subjectLength - 1, -1, -1):
        for j in range(subjectLength - 1, -1, -1):
            if i != j:
                check, k = myhog.checkIntersections(positionList[i],positionList[j],probaList[i],probaList[j],minOverlap_x,minOverlap_y)
                if check is True:
                    if k == 0:
                        fuoriList.append(i)
                    else:
                        fuoriList.append(j)

    definitiveList = []
    defproba = []
    for i in range(0,len(positionList)):
        if i not in fuoriList:
            definitiveList.append(positionList[i])
            defproba.append(probaList[i])

    myhog.draw_detections(framergb, definitiveList, defproba, matching_threshold, 2)

    file = open(pathTest+os.path.basename(nameFrame)+'.gt','w')
    for f in definitiveList:
       file.write(str(f[0]) + ' ' + str(f[1]) + ' ' + str(f[2]) + ' ' + str(f[3]) + '\n')
    file.close()

    if REALTIME is False:
        framebuf.append(framergb)

    return totalRect, len(definitiveList)
########################################################################################
# program start here
def global_var(realtime, verbose, batch):
    global REALTIME
    global VERBOSE
    global BATCH

    REALTIME = realtime
    VERBOSE = verbose
    BATCH = batch
    return

def hog_parameters(st=12, mt=0.9, sc=1.0):
    global step
    global orient
    global matching_threshold
    global scale
    global frameScale
    global minOverlap_x
    global minOverlap_y
    # di quanto si muove la finestra di scorrimento
    step = st
    # soglia di match tra le finestre
    matching_threshold = mt
    # scala della finetra di ritaglio
    scale = sc

    return

def mask_param():
    # soglia da maschera da computare o meno
    mask_threshold = 0.3

    return

seqList = {
    "Set_3/ID_76/Camera_1/Seq_1/",
    "Set_3/ID_80/Camera_1/Seq_1/",
    "Set_4/ID_112/Camera_8/Seq_1/",
    "Set_4/ID_114/Camera_8/Seq_1/",
    "Set_4/ID_115/Camera_8/Seq_1/",
    "Set_4/ID_121/Camera_8/Seq_1/",
    "Set_4/ID_122/Camera_8/Seq_1/",
    "Set_4/ID_122/Camera_8/Seq_3/",
    "Set_4/ID_123/Camera_8/Seq_1/",
    "Set_4/ID_129/Camera_8/Seq_1/",
    "Set_4/ID_139/Camera_8/Seq_1/"
}

def batch():
    stepList = [11, 13, 15]
    mtList = [0.9, 0.95, 0.99]
    scalList = [0.9, 1.0, 1.1]

    print "batch start"
    for s in stepList:
        for m in mtList:
            for c in scalList:
                accList= []
                recList = []
                precList = []
                f1List = []
                timeList = []
                out = ""
                filename = pathSave + "log-s" + str(s) + "-m" + str(m) + "-c" + str(c) + ".txt"
                print "start " + str(s) + " " + str(m) + " " + str(c)
                for l in seqList:
                    hog_parameters(s, m, c)
                    accuracy, recall, precision, f1score, time = main(l)
                    out += "Set " + str(l) + "\n"
                    out += "Accuracy = "+str(accuracy) + " "
                    out += "Recall = "+str(recall) + " "
                    out += "Precision = "+str(precision) + " "
                    out += "F1 Score = "+str(f1score) + " "
                    out += "Total Time = " + str(time)
                    out += "\n"
                    accList.append(accuracy)
                    recList.append(recall)
                    precList.append(precision)
                    f1List.append(f1score)
                    timeList.append(time)

                out += "Mean Avg Accuracy: " + str(np.mean(accList)) + "\n"
                out += "Mean Avg Recall: " + str(np.mean(recList)) + "\n"
                out += "Mean Avg Precision: " + str(np.mean(precList)) + "\n"
                out += "Mean Avg F1: " + str(np.mean(f1List)) + "\n"
                out += "Mean Elapsed Time:" + str(np.mean(timeList)) + "\n"
                with open(filename, "w") as f:
                    f.write(out)

    print "team bottle wins"
    return

main(pathFolder)