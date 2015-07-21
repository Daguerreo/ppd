__author__ = 'Roberta'

import DatasetManager
import numpy as np

# se true, spamma tutte le aree e intersezioni trovati
SHOWSTAT = False

class Statistic:
    def __init__(self):
        pass

    def calcPositiveNegative(self, pathGT, path, windowscale, threshold):
        dm = DatasetManager.DatasetManager()

        TP = 0
        TN = 0
        FP = 0
        FN = 0

        listGT = dm.getFigNames(pathGT, False)
        # listGT = lista dei basename dei frame
        listTest = dm.getFigNames(path, False)

        # scorri lista GT e per ogni elemento della lista cerca se c'e' lo stesso frame nella lista del Test
        for i in range(len(listGT)):
            # se trova il corrispettivo vuol dire che in un frame dove veramente c'erano persone,
            # HOG ne ha trovata almeno una

            # apre il file di testo e accede ad ogni riga = ogni riga c'e' un rect
            fileGT = open(pathGT + listGT[i], 'r')
            fileTest = open(path + listTest[i], 'r')

            fGT = fileGT.readlines()
            fTest = fileTest.readlines()

            if not fGT and not fTest:
                TN += 1
            elif fGT and fTest:
                flag1 = False

                # per ogni riga di fileGT confronta con tutte le righe di fileTest
                for s in fGT:
                    sl = s.split(' ')
                    # rectGT = (int(sl[0]), int(sl[1]),int(sl[0])+int(sl[2]),  int(sl[1])+int(sl[3]))
                    a = int(int(sl[0]) * windowscale)
                    b = int(int(sl[1]) * windowscale)
                    c = int(int(sl[0]) * windowscale) + int(int(sl[2]) * windowscale)
                    d = int(int(sl[1]) * windowscale) + int(int(sl[3]) * windowscale)
                    rectGT = (a, b, c, d)
                    for c in fTest:
                        cl = c.split(' ')
                        rectTest = (int(cl[0]), int(cl[1]), int(cl[2]), int(cl[3]))

                        # print rectGT
                        # print rectTest

                        # frame=frames
                        # cv2.rectangle(frame, (rectGT[0], rectGT[1]), (rectGT[2], rectGT[3]), (0, 0, 255), 3)
                        # cv2.rectangle(frame, (rectTest[0], rectTest[1]), (rectTest[2], rectTest[3]), (255, 0, 0), 3)
                        # cv2.imshow("ciccio",frame)
                        # cv2.waitKey()
                        # calcola le coordinate dell'intersezione
                        # x3 = max(rectGT[0],rectTest[0])
                        # y3 = max(rectGT[1],rectTest[1])

                        # x4 = min(rectGT[2],rectTest[2])
                        # y4 = min(rectGT[3],rectTest[3])

                        # calcola le aree
                        # blockArea = float(np.abs(x3-x4)*np.abs(y3-y4))
                        # area1 = float(np.abs(rectGT[2]-rectGT[0])*np.abs(rectGT[3]-rectGT[1]))
                        # area2 = float(np.abs(rectTest[2]-rectTest[0])*np.abs(rectTest[3]-rectTest[1]))

                        # percentuale di intersezione
                        # overlap=blockArea/(area1+area2+blockArea)
                        # overlap = math.hypot(rectGT[0]-rectTest[0],rectGT[1]-rectTest[1])
                        overlap = np.sqrt((rectGT[0] - rectTest[0]) ** 2 + (rectGT[1] - rectTest[1]) ** 2)
                        # se c'e' abbastanza intersezione vuol dire che i due rect corrispondono
                        if overlap < threshold:
                            flag1 = True

                    if flag1 is False:
                        FN += 1
                    else:
                        TP += 1

                flag2 = False

                # per ogni riga di fileTest confronta con tutte le righe di fileGT:
                # i true positive li abbiamo gia calcolati

                for c in fTest:
                    cl = c.split(' ')
                    rectTest = (int(cl[0]), int(cl[1]), int(cl[2]), int(cl[3]))
                    for s in fGT:
                        sl = s.split(' ')
                        a = int(int(sl[0]) * windowscale)
                        b = int(int(sl[1]) * windowscale)
                        c = int(int(sl[0]) * windowscale) + int(int(sl[2]) * windowscale)
                        d = int(int(sl[1]) * windowscale) + int(int(sl[3]) * windowscale)
                        rectGT = (a, b, c, d)


                        # x3=max(rectGT[0],rectTest[0])
                        # y3=max(rectGT[1],rectTest[1])

                        # x4=min(rectGT[2],rectTest[2])
                        # y4=min(rectGT[3],rectTest[3])

                        # blockArea=float(np.abs(x3-x4)*np.abs(y3-y4))
                        # area1=float(np.abs(rectGT[2]-rectGT[0])*np.abs(rectGT[3]-rectGT[1]))
                        # area2=float(np.abs(rectTest[2]-rectTest[0])*np.abs(rectTest[3]-rectTest[1]))

                        # overlap=blockArea/(area1+area2+blockArea)
                        # overlap=math.hypot(rectGT[0]-rectTest[0],rectGT[1]-rectTest[1])
                        overlap = np.sqrt((rectGT[0] - rectTest[0]) ** 2 + (rectGT[1] - rectTest[1]) ** 2)
                        if overlap < threshold:
                            flag2 = True

                    if flag2 is False:
                        FP += 1

            elif fGT and not fTest:
                fileGT = open(pathGT + listGT[i], 'r')
                fGT = fileGT.readlines()

                FN += len(fGT)

            else:
                FP += len(fTest)

        return TP, TN, FP, FN

    def calcPerformance(self, TP, TN, FP, FN):
        TP = float(TP)
        TN = float(TN)
        FP = float(FP)
        FN = float(FN)

        accuracy = (TP + TN) / (TN + TP + FN + FP)
        recall = TP / (TP + FN)
        f1score = 2 * TP / (2 * TP + FP + FN)

        if TP == 0 and FP == 0:
            precision = 0.0
        else:
            precision = TP / (TP + FP)

        return accuracy, recall, precision, f1score

    def calcStatisticsValues(self, pathGT, path, windowscale, threshold):
        dm = DatasetManager.DatasetManager()

        TP = 0
        TN = 0
        FP = 0
        FN = 0

        listGT = dm.getFigNames(pathGT, False)
        # listGT=lista dei basename dei frame
        listTest = dm.getFigNames(path, False)

        # scorri lista GT e per ogni elemento della lista cerca se c'e' lo stesso frame nella lista del Test
        for i in range(len(listGT)):
            if listGT[i] in listTest:
                # se trova il corrispettivo vuol dire che in un frame dove veramente c'erano persone,
                # HOG ne ha trovata almeno una

                # apre il file di testo e accede ad ogni riga = ogni riga c'e' un rect
                fileGT = open(pathGT + listGT[i], 'r')

                k = listTest.index(listGT[i])
                fileTest = open(path + listTest[k], 'r')

                fGT = fileGT.readlines()
                fTest = fileTest.readlines()
                flag1 = False

                # per ogni riga di fileGT confronta con tutte le righe di fileTest
                for s in fGT:
                    sl = s.split(' ')
                    a = int(int(sl[0]) * windowscale)
                    b = int(int(sl[1]) * windowscale)
                    c = int(int(sl[0]) * windowscale) + int(int(sl[2]) * windowscale)
                    d = int(int(sl[1]) * windowscale) + int(int(sl[3]) * windowscale)
                    rectGT = (a, b, c, d)
                    # rectGT=(int(sl[0]),int(sl[1]),int(sl[2]),int(sl[3]))
                    for c in fTest:
                        cl = c.split(' ')
                        rectTest = (int(cl[0]), int(cl[1]), int(cl[2]), int(cl[3]))

                        # calcola le coordinate dell'intersezione
                        x3 = max(rectGT[0], rectTest[0])
                        y3 = max(rectGT[1], rectTest[1])

                        x4 = min(rectGT[2], rectTest[2])
                        y4 = min(rectGT[3], rectTest[3])

                        # calcola le aree
                        blockArea = float(np.abs(x3 - x4) * np.abs(y3 - y4))
                        area1 = float(np.abs(rectGT[2] - rectGT[0]) * np.abs(rectGT[3] - rectGT[1]))
                        area2 = float(np.abs(rectTest[2] - rectTest[0]) * np.abs(rectTest[3] - rectTest[1]))
                        if SHOWSTAT is True:
                            print "blockArea: " + str(blockArea)
                            print "area1: " + str(area1)
                            print "area2: " + str(area2)
                        # percentuale di intersezione
                        overlap = blockArea / (area1 + area2 - blockArea)
                        # overlap=math.hypot(rectGT[0]-rectTest[0],rectGT[1]-rectTest[1])
                        if SHOWSTAT is True:
                            print "overlap: " + str(overlap)
                        # se c'e' abbastanza intersezione vuol dire che i due rect corrispondono
                        if overlap > threshold:
                            flag1 = True

                    if flag1 is False:
                        FN += 1
                    else:
                        TP += 1

                flag2 = False

                # per ogni riga di fileTest confronta con tutte le righe di fileGT:
                # i true positive li abbiamo gia calcolati

                for c in fTest:
                    cl = c.split(' ')
                    rectTest = (int(cl[0]), int(cl[1]), int(cl[2]), int(cl[3]))
                    for s in fGT:
                        sl = s.split(' ')
                        a = int(int(sl[0]) * windowscale)
                        b = int(int(sl[1]) * windowscale)
                        c = int(int(sl[0]) * windowscale) + int(int(sl[2]) * windowscale)
                        d = int(int(sl[1]) * windowscale) + int(int(sl[3]) * windowscale)
                        rectGT = (a, b, c, d)

                        x3 = max(rectGT[0], rectTest[0])
                        y3 = max(rectGT[1], rectTest[1])

                        x4 = min(rectGT[2], rectTest[2])
                        y4 = min(rectGT[3], rectTest[3])

                        blockArea = float(np.abs(x3 - x4) * np.abs(y3 - y4))
                        area1 = float(np.abs(rectGT[2] - rectGT[0]) * np.abs(rectGT[3] - rectGT[1]))
                        area2 = float(np.abs(rectTest[2] - rectTest[0]) * np.abs(rectTest[3] - rectTest[1]))

                        overlap = blockArea / (area1 + area2 - blockArea)
                        # overlap=math.hypot(rectGT[0]-rectTest[0],rectGT[1]-rectTest[1])

                        if overlap > threshold:
                            flag2 = True

                    if flag2 is False:
                        FP += 1

            else:
                fileGT = open(pathGT + listGT[i], 'r')
                fGT = fileGT.readlines()

                FN += len(fGT)

        for i in range(len(listTest)):
            if listTest[i] not in listGT:
                fileTest = open(path + listTest[i], 'r')
                fTest = fileTest.readlines()

                FP += len(fTest)

        return TP, TN, FP, FN
