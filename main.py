from cv2 import VideoCapture
import numpy as np
import cv2
import pyttsx3

thres = 0.5 #Threshold to Detect Object
cap = cv2.VideoCapture(1)
cap.set(3,400)
cap.set(4,600)

classNames = []
classFile='E:/Artificial Intelligence/Object Detection SSD/coco2.names'
with open (classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split ('\n')

configPath = 'E:/Artificial Intelligence/Object Detection SSD/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'E:/Artificial Intelligence/Object Detection SSD/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=0.5)
    #print(classIds,bbox)

    if len(classIds)!= 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box, color=(0,255,0), thickness=2)
            cv2.putText(img,classNames[classId-1],(box[0]+10, box[1]+30),
                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.putText(img,str(round(confidence*100.2)),(box[0]+200, box[1]+30),
                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    
    engine=pyttsx3.init()
    engine.say("Awas Ada")
    engine.say(classNames[classId-1])
    engine.runAndWait()

    cv2.imshow("Output", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()