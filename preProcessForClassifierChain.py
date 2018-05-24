from darkflow.net.build import TFNet
import cv2
import math
import os

options = {"model": "./cfg/yolo.cfg",
           "load": "./bin/yolo.weights",
           "threshold": 0.3}

tfnet = TFNet(options)

labelOfInterest = 'dog'

videoDirectory = "./data/dataVideo_dogs/"
outputDirectory = "./data/results_with_dogs/"

for filename in os.listdir(videoDirectory):

    videoFile = videoDirectory + filename
    videoIn = cv2.VideoCapture(videoFile) # dummy video atm
    frameRate = videoIn.get(5)

    maxConfidenceForLOI = 0.0;

    print("Making predictions for video: " + videoFile)

    # while the video still has frames
    while(videoIn.isOpened()):
        frameId = videoIn.get(1) # current frame number

        ret, frame = videoIn.read()

        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0): # every second
            result = tfnet.return_predict(frame)

            for objectDetected in result:
                if objectDetected['label'] == labelOfInterest:
                    if objectDetected['confidence'] > maxConfidenceForLOI:
                        maxConfidenceForLOI = objectDetected['confidence']
                    print(objectDetected['label'] + ": " + str(objectDetected['confidence']))
    videoIn.release()
    f = open(outputDirectory + filename + ".txt", "w+")
    f.write(str(maxConfidenceForLOI))
    print("THE MAX CONFIDENCE FOR A DOG IN THIS VIDEO " + videoFile + " is: " + str(maxConfidenceForLOI))


# tfnet = TFNet(options)
#
# imgcv = cv2.imread("./sample_img/sample_dog.jpg")
# result = tfnet.return_predict(imgcv)
# print(result)

