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

coco_keys = [
    'person',
    'bicycle',
    'car',
    'motorbike',
    'aeroplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'sofa',
    'pottedplant',
    'bed',
    'diningtable',
    'toilet',
    'tvmonitor',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush'
]

nonempty_dict = dict.fromkeys(coco_keys, 0.0)

print(nonempty_dict)

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


