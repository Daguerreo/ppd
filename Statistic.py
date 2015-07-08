__author__ = 'Chiara'

import os
import DatasetManager
import numpy as np
import math


class Statistic:

    def __init__(self):
        pass

    def calcAccuracy(self,pathGT,path, threshold):
        dm=DatasetManager.DatasetManager()

        TRUE_POSITIVE=0
        TRUE_NEGATIVE=0
        FALSE_POSITIVE=0
        FALSE_NEGATIVE=0


        listGT=dm.getFigNames(pathGT,False)
        #listGT=lista dei basename dei frame
        listTest=dm.getFigNames(path,False)

        #scorri lista GT e per ogni elemento della lista cerca se c'e' lo stesso frame nella lista del Test
        for i in range(len(listGT)):
            #se trova il corrispettivo vuol dire che in un frame dove veramente c'erano persone, HOG ne ha trovata almeno una

            #apre il file di testo e accede ad ogni riga = ogni riga c'e' un rect
            fileGT=open('groundtruth/'+listGT[i],'r')
            fileTest=open('rects/'+listTest[i],'r')

            fGT=fileGT.readlines()
            fTest=fileTest.readlines()

            if not fGT and not fTest:
                TRUE_NEGATIVE +=1
            elif fGT and fTest:


                flag1=False


                #per ogni riga di fileGT confronta con tutte le righe di fileTest
                for s in fGT:
                    sl=s.split(' ')
                    rectGT=(int(sl[0]),int(sl[1]),int(sl[2]),int(sl[3]))
                    for c in fTest:
                        cl=c.split(' ')
                        rectTest=(int(cl[0]),int(cl[1]),int(cl[2]),int(cl[3]))

                        #calcola le coordinate dell'intersezione
                        x3=max(rectGT[0],rectTest[0])
                        y3=max(rectGT[1],rectTest[1])

                        x4=min(rectGT[2],rectTest[2])
                        y4=min(rectGT[3],rectTest[3])

                        #calcola le aree
                        blockArea=float(np.abs(x3-x4)*np.abs(y3-y4))
                        area1=float(np.abs(rectGT[2]-rectGT[0])*np.abs(rectGT[3]-rectGT[1]))
                        area2=float(np.abs(rectTest[2]-rectTest[0])*np.abs(rectTest[3]-rectTest[1]))

                        #percentuale di intersezione
                        #overlap=blockArea/(area1+area2+blockArea)
                        overlap=math.hypot(rectGT[0]-rectTest[0],rectGT[1]-rectTest[1])

                        #se c'e' abbastanza intersezione vuol dire che i due rect corrispondono
                        if overlap > threshold:
                            flag1=True

                    if flag1 is False:
                        FALSE_NEGATIVE +=1
                    else:
                        TRUE_POSITIVE +=1

                flag2=False

                #per ogni riga di fileTest confronta con tutte le righe di fileGT: i true positive li abbiamo gia calcolati

                for c in fTest:
                    cl=c.split(' ')
                    rectTest=(int(cl[0]),int(cl[1]),int(cl[2]),int(cl[3]))
                    for s in fGT:
                        sl=s.split(' ')
                        rectGT=(int(sl[0]),int(sl[1]),int(sl[2]),int(sl[3]))

                        x3=max(rectGT[0],rectTest[0])
                        y3=max(rectGT[1],rectTest[1])

                        x4=min(rectGT[2],rectTest[2])
                        y4=min(rectGT[3],rectTest[3])

                        blockArea=float(np.abs(x3-x4)*np.abs(y3-y4))
                        area1=float(np.abs(rectGT[2]-rectGT[0])*np.abs(rectGT[3]-rectGT[1]))
                        area2=float(np.abs(rectTest[2]-rectTest[0])*np.abs(rectTest[3]-rectTest[1]))

                        #overlap=blockArea/(area1+area2+blockArea)
                        overlap=math.hypot(rectGT[0]-rectTest[0],rectGT[1]-rectTest[1])


                        if overlap >threshold:
                            flag2=True

                    if flag2 is False:
                        FALSE_POSITIVE += 1

            elif fGT and not fTest:
                fileGT=open('groundtruth/'+listGT[i],'r')
                fGT=fileGT.readlines()

                FALSE_NEGATIVE += len(fGT)

            else:
                FALSE_POSITIVE += len(fTest)

        return TRUE_POSITIVE,TRUE_NEGATIVE,FALSE_POSITIVE,FALSE_NEGATIVE



    def calcAccuracy_per_i_posteri(self,pathGT,path, threshold):
        dm=DatasetManager.DatasetManager()

        TRUE_POSITIVE=0
        TRUE_NEGATIVE=0
        FALSE_POSITIVE=0
        FALSE_NEGATIVE=0

        
        listGT=dm.getFigNames(pathGT,False)
        #listGT=lista dei basename dei frame
        listTest=dm.getFigNames(path,False)

        #scorri lista GT e per ogni elemento della lista cerca se c'e' lo stesso frame nella lista del Test
        for i in range(len(listGT)):
            if listGT[i] in listTest:
                #se trova il corrispettivo vuol dire che in un frame dove veramente c'erano persone, HOG ne ha trovata almeno una

                #apre il file di testo e accede ad ogni riga = ogni riga c'e' un rect
                fileGT=open('groundtruth/'+listGT[i],'r')

                k=listTest.index(listGT[i])
                fileTest=open('rects/'+listTest[k],'r')

                fGT=fileGT.readlines()
                fTest=fileTest.readlines()
                flag1=False

                #per ogni riga di fileGT confronta con tutte le righe di fileTest
                for s in fGT:
                    sl=s.split(' ')
                    rectGT=(int(sl[0]),int(sl[1]),int(sl[2]),int(sl[3]))
                    for c in fTest:
                        cl=c.split(' ')
                        rectTest=(int(cl[0]),int(cl[1]),int(cl[2]),int(cl[3]))

                        #calcola le coordinate dell'intersezione
                        x3=max(rectGT[0],rectTest[0])
                        y3=max(rectGT[1],rectTest[1])

                        x4=min(rectGT[2],rectTest[2])
                        y4=min(rectGT[3],rectTest[3])

                        #calcola le aree
                        blockArea=float(np.abs(x3-x4)*np.abs(y3-y4))
                        area1=float(np.abs(rectGT[2]-rectGT[0])*np.abs(rectGT[3]-rectGT[1]))
                        area2=float(np.abs(rectTest[2]-rectTest[0])*np.abs(rectTest[3]-rectTest[1]))

                        #percentuale di intersezione
                        #overlap=blockArea/(area1+area2+blockArea)
                        overlap=math.hypot(rectGT[0]-rectTest[0],rectGT[1]-rectTest[1])

                        #se c'e' abbastanza intersezione vuol dire che i due rect corrispondono
                        if overlap > threshold:
                            flag1=True

                    if flag1 is False:
                        FALSE_NEGATIVE +=1
                    else:
                        TRUE_POSITIVE +=1

                flag2=False

                #per ogni riga di fileTest confronta con tutte le righe di fileGT: i true positive li abbiamo gia calcolati

                for c in fTest:
                    cl=c.split(' ')
                    rectTest=(int(cl[0]),int(cl[1]),int(cl[2]),int(cl[3]))
                    for s in fGT:
                        sl=s.split(' ')
                        rectGT=(int(sl[0]),int(sl[1]),int(sl[2]),int(sl[3]))

                        x3=max(rectGT[0],rectTest[0])
                        y3=max(rectGT[1],rectTest[1])

                        x4=min(rectGT[2],rectTest[2])
                        y4=min(rectGT[3],rectTest[3])

                        blockArea=float(np.abs(x3-x4)*np.abs(y3-y4))
                        area1=float(np.abs(rectGT[2]-rectGT[0])*np.abs(rectGT[3]-rectGT[1]))
                        area2=float(np.abs(rectTest[2]-rectTest[0])*np.abs(rectTest[3]-rectTest[1]))

                        #overlap=blockArea/(area1+area2+blockArea)
                        overlap=math.hypot(rectGT[0]-rectTest[0],rectGT[1]-rectTest[1])


                        if overlap >threshold:
                            flag2=True

                    if flag2 is False:
                        FALSE_POSITIVE += 1

            else:
                fileGT=open('groundtruth/'+listGT[i],'r')
                fGT=fileGT.readlines()

                FALSE_NEGATIVE += len(fGT)

        for i in range(len(listTest)):
            if listTest[i] not in listGT:
                fileTest=open('rects/'+listTest[i],'r')
                fTest=fileTest.readlines()

                FALSE_POSITIVE += len(fTest)

        return TRUE_POSITIVE,TRUE_NEGATIVE,FALSE_POSITIVE,FALSE_NEGATIVE

















