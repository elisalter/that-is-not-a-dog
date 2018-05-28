from darkflow.net.build import TFNet
import cv2
import math
import os

options = {"model": "./cfg/yolo.cfg",
           "load": "./bin/yolo.weights",
           "threshold": 0.0}

tfnet = TFNet(options)

videoDirectory = "./data/dataVideo_noDogs/"
outputDirectory = "./data/multi_results_without_dogs/"

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


for filename in os.listdir(videoDirectory):

    videoFile = videoDirectory + filename
    videoIn = cv2.VideoCapture(videoFile) # dummy video atm
    frameRate = videoIn.get(5)

    max_pred_dict = dict.fromkeys(coco_keys, 0.0)

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
                currentMax = max_pred_dict[objectDetected['label']]
                if objectDetected['confidence'] > currentMax:
                    max_pred_dict[objectDetected['label']] = objectDetected['confidence']


    videoIn.release()
    f = open(outputDirectory + filename + ".txt", "w+")
    f.write(str(list(max_pred_dict.values())))
    print(max_pred_dict)
    print("-------------------------------------------------------------")


