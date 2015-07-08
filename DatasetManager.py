__author__ = 'Daguerreo'

import Sample
import os
import random
import pickle

class DatasetManager:
    def __init__(self):
        pass

    def getFigNames(self,path):
        name_fig=[]

        for s in os.listdir(path):
            filename = os.path.basename(s)
            if os.path.splitext(filename)[1] == '.jpg' or os.path.splitext(filename)[1] == '.bmp':
                name_fig.append(s)
        return name_fig

    def getDataset(self, path):
        name_fig=[]
        label=[]

        categories=os.listdir(path)
        for c in range(len(categories)):
            for s in os.listdir(os.path.join(path,categories[c])):
                filename = os.path.basename(s)
                str = os.path.splitext(filename)[1]
                if os.path.splitext(filename)[1] == '.jpg' or os.path.splitext(filename)[1] == '.bmp':

                    name_fig.append(os.path.join(path,categories[c],s))
                    label.append(c)

        return name_fig,label

    def sortDataset(self, path, test_percentage, do_shuffle=False):
        # percentage definisce il numero di immagini da prendere per il train
        testSet = []
        trainingSet = []
        categories=os.listdir(path)
        # accede alle cartelle
        for c in range(len(categories)):
            imgName=[]
            # singolo nome delle immagini
            for s in os.listdir(os.path.join(path,categories[c])):
                filename = os.path.basename(s)
                str = os.path.splitext(filename)[1]
                if os.path.splitext(filename)[1] == '.jpg' or os.path.splitext(filename)[1] == '.bmp':
                    # prendiamo tutte le immagini della sottocartella
                    imgName.append(os.path.join(path,categories[c],s))

            if do_shuffle is True:
                random.shuffle(imgName)

            # calcola la percentuale di dataset da prendere
            n = int(len(imgName)*test_percentage)
            for l in range(0,n):
                # crea il sample e associa i parametri
                sample = Sample.Sample()
                sample.label = categories[c]
                sample.path = imgName[l]
                testSet.append(sample)

            for l in range(n,len(imgName)):
                sample = Sample.Sample()
                sample.label = categories[c]
                sample.path = imgName[l]
                trainingSet.append(sample)

        return trainingSet, testSet

