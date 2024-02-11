import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
#classifier module
#hand detection library
import numpy as np
import math
import time


cap=cv2.VideoCapture(0)
# to capture web cam and 0 is default value of web cam
detector=HandDetector(maxHands=2) #detect total of 1 hand only to keep it simple
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset=20
imageSize = 300

folder="Data/C" #images when pressed s will be saved here
counter=0
labels = ["A", "B", "C", "D","E","1","2","3","Love","Enjoy","Fever","Washroom"]

while True:
    success, img=cap.read()
    imgCopy = img.copy()
    hands, img=detector.findHands(img)#it will detect hand when web cam opens
    if hands:
        hand=hands[0]#means only one hand , if max hands was 2 then hands[1] means second hand
        x,y,w,h=hand['bbox']#gives dimensions of hand

        #create image by ourselves with white background and then oervlay the image on that white bacground and then resize and centre it so thats it fits the white box
        #created from matrix
        imageWhite = np.ones((imageSize,imageSize,3), np.uint8)*255
        #square matrix of 300 by 300 and 3 means colored image i.e r,g,b
        #also give range of values and in our case its from 0 to 255 so iuse unsigned integer
        imgCrop = img[y-offset:y+h+offset , x-offset:x+w+offset]#gives spaces for hand images

        imgCropShape=imgCrop.shape


        #image resiszing of height so that it fits the white background
        #if height>width ,then make hight =300
        # if width>height ,then stretch width to 300

        ascpetRatio = h/w


        #for height
        if ascpetRatio>1: #means height is bigger
            k=imageSize/h
            wCal=math.ceil(k*w)
            imgResize=cv2.resize(imgCrop,(wCal,imageSize))
            imageResizeShape=imgResize.shape
            #now to centre the image
            wGap=math.ceil((imageSize-wCal)/2)
            imageWhite[:,wGap:wCal+wGap] = imgResize  # this is gonna lay our image on white background
            # 0-starting point of height , imgCropShape[0]-end point of height and same for width
            predicition, index=classifier.getPrediction(imageWhite, draw=False)#get predicition gives two values one predicition and one index
            print(predicition , index)



        #for width
        else:
            k = imageSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imageSize, hCal))
            imageResizeShape = imgResize.shape
            # now to centre the image
            hGap = math.ceil((imageSize - hCal) / 2)
            imageWhite[hGap:hCal + hGap,:] = imgResize
            predicition, index = classifier.getPrediction(imageWhite, draw=False)

        cv2.rectangle(imgCopy, (x - offset, y - offset-50), (x - offset+90 , y - offset-50+50),(255,0,255), cv2.FILLED)
        cv2.putText(imgCopy, labels[index], (x,y-25), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgCopy ,(x-offset,y-offset), (x+w+offset,y+h+offset), (255,0,255),4)

        cv2.imshow("ImageCrop" , imgCrop)
        cv2.imshow("ImageWhite", imageWhite)


    cv2.imshow("Image" , imgCopy) #open webcam and title of tab will be image
    cv2.waitKey(1) #wait of 1 milisecond


