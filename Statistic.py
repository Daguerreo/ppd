__author__ = 'Chiara'

import os
import DatasetManager
import numpy as np


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
        listTest=dm.getFigNames(path,False)


        for i in range(len(listGT)):
            if listGT[i] in listTest:
                fileGT=open('groundtruth76/'+listGT[i],'r')

                k=listTest.index(listGT[i])
                fileTest=open('rects/'+listTest[k],'r')

                fGT=fileGT.readlines()
                fTest=fileTest.readlines()
                flag1=False

                for s in fGT:
                    sl=s.split(' ')
                    rectGT=(int(sl[0]),int(sl[1]),int(sl[2]),int(sl[3]))
                    for c in fTest:
                        cl=c.split(' ')
                        rectTest=(int(cl[0]),int(cl[1]),int(cl[2]),int(cl[3]))

                        x3=max(rectGT[0],rectTest[0])
                        y3=max(rectGT[1],rectTest[1])

                        x4=min(rectGT[2],rectTest[2])
                        y4=min(rectGT[3],rectTest[3])

                        blockArea=float(np.abs(x3-x4)*np.abs(y3-y4))
                        area1=float(np.abs(rectGT[2]-rectGT[0])*np.abs(rectGT[3]-rectGT[1]))
                        area2=float(np.abs(rectTest[2]-rectTest[0])*np.abs(rectTest[3]-rectTest[1]))

                        overlap=blockArea/(area1+area2+blockArea)

                        if overlap > threshold:
                            flag1=True

                    if flag1 is False:
                        FALSE_NEGATIVE +=1
                    else:
                        TRUE_POSITIVE +=1
                flag2=False
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

                        overlap=blockArea/(area1+area2+blockArea)

                        if overlap >threshold:
                            flag2=True

                    if flag2 is False:
                        FALSE_POSITIVE += 1

            else:
                fileGT=open('groundtruth76/'+listGT[i],'r')
                fGT=fileGT.readlines()

                FALSE_NEGATIVE += len(fGT)

        for i in range(len(listTest)):
            if listTest[i] not in listGT:
                fileTest=open('rects/'+listTest[i],'r')
                fTest=fileTest.readlines()

                FALSE_POSITIVE += len(fTest)

        return TRUE_POSITIVE,TRUE_NEGATIVE,FALSE_POSITIVE,FALSE_NEGATIVE

















