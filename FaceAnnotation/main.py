# import the necessary packages

import cv2
import os
from glob import glob
import numpy as np
import random

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
image=np.array([])
clone=np.array([])
rand_index=237

def setupFolders():
    if not os.path.isdir('groundtruth'):
        os.mkdir('groundtruth')
    if not os.path.isdir('img_benchmark'):
        os.mkdir('img_benchmark')

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping, image

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN and not cropping:
        refPt = [(x, y)]
        cropping = True

    if event == cv2.EVENT_LBUTTONDOWN and cropping:
        cp=image.copy()
        cv2.rectangle(cp, refPt[0], (x,y), (0,255,0), 2)
        cv2.imshow("image", cp)

    # check to see if the left mouse button was released
    if event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False
        # draw a rectangle around the region of interest
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

if __name__=="__main__":
    setupFolders()
    folder="C:/Users/Chiara/Desktop/MuMet/Set_4/ID_115/Camera_8/Seq_1/"
    images=[y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.jpg'))]

    key=0
    faces=[]
    # load the image and setup the mouse callback function
    #rand_index=np.arange(0,len(images)) #random.randint(0,len(images)-1)
    image = cv2.imread(images[rand_index])
    clone=image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    annotated=0
    while key!=ord('q'):

        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey()

        if key==32: #spacebar
            cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
            faces.append(np.array([refPt[0][0], refPt[0][1], refPt[1][0]-refPt[0][0], refPt[1][1]-refPt[0][1]]))

        elif key==ord('p'): #pass
            faces=[]
            # load the image and setup the mouse callback function
            #rand_index=random.randint(0,len(images)-1)
            rand_index=rand_index+1
            image = cv2.imread(images[rand_index])
            clone=image.copy()
            cv2.namedWindow("image")
            cv2.setMouseCallback("image", click_and_crop)

        elif key==13:
            file=open('groundtruth/'+images[rand_index].split('\\',1)[1]+'.gt','w')
            for f in faces:
                file.write(str(f[0]) + ' ' + str(f[1]) + ' ' + str(f[2]) + ' ' + str(f[3]) + '\n')
            file.close()
            faces=[]
            cv2.imwrite('img_benchmark/'+images[rand_index].split('\\',1)[1],clone)
            annotated+=1
            print 'Number of annotated images: ' + str(annotated)
            #rand_index=random.randint(0,len(images)-1)
            rand_index=rand_index+1

            image = cv2.imread(images[rand_index])
            clone=image.copy()
            cv2.namedWindow("image")
            cv2.setMouseCallback("image", click_and_crop)

