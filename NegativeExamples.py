__author__ = 'User'

import cv2
import DatasetManager
import Util

def NegativeExamples(path,step,a,b,x,y,savepath):

    st = int(step)
    dm = DatasetManager.DatasetManager()
    set = dm.getDataset(path)[0]

    for k in range (a,b):
        img = cv2.imread(set[k],cv2.IMREAD_COLOR)
        for i in range(0,img.shape[0]-y,st):
            for j in range(0,img.shape[1]-x,st):
                if j+x<img.shape[1] and i+y<img.shape[0]:
                    window = img[i:i+y,j:j+x]
                    cv2.imwrite(savepath + "img_" + str(j) + str(i)+ ".jpg",window)

    return

step = 10
x=48
y=128
path = "sequences/Set_4/ID_129/Camera_8/"

a = 10
b = 20

util = Util.Util()
util.delFolderContent(path)
NegativeExamples(path,step,a,b,x,y,"cut/")