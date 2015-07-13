import os
import StatIstance
from collections import Counter

path = "log/"
statList = []

def getAcc(si):
    return si.accuracy

def getRec(si):
    return si.recall

def getPre(si):
    return si.precision

def getF1(si):
    return si.f1Score

for i in os.listdir(path):
    name = ""
    filename = os.path.basename(i)
    if os.path.splitext(filename)[1] == '.txt':
        name = os.path.splitext(filename)[0]

    file = open(path + filename, 'r')
    lines = file.readlines()

    sep = lines[-1].split(',')
    stat = StatIstance.StatIstance()
    stat.name = name
    stat.accuracy = sep[0]
    stat.recall = sep[1]
    stat.precision = sep[2]
    stat.f1Score = sep[3]
    stat.totalTime = sep[4]
    statList.append(stat)

statList = sorted(statList, key=getPre, reverse=True)
labelList = []

for i in range(0,10):
    print statList[i].name + " Precision: " + statList[i].precision + " Time " + statList[i].totalTime
    labelList.append(statList[i].name)

statList = sorted(statList, key=getAcc, reverse=True)
print ""
for i in range(0,10):
    print statList[i].name + " Accuracy: " + statList[i].accuracy + " Time " + statList[i].totalTime
    labelList.append(statList[i].name)

statList = sorted(statList, key=getRec, reverse=True)
print ""
for i in range(0,10):
    print statList[i].name + " Recall: " + statList[i].recall + " Time " + statList[i].totalTime
    labelList.append(statList[i].name)

statList = sorted(statList, key=getF1, reverse=True)
print ""
for i in range(0,10):
    print statList[i].name + " F1 Score: " + statList[i].f1Score + " Time " + statList[i].totalTime
    labelList.append(statList[i].name)
print ""
print Counter(labelList)