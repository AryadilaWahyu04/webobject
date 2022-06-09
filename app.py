from cv2 import VideoCapture
import numpy as np
import cv2
import pyttsx3
from flask import Flask,render_template, Response, url_for
import time

app=Flask(__name__)

thres = 0.5 #Threshold to Detect Object
camera = cv2.VideoCapture(0)
camera.set(3,400)
camera.set(4,600)


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

def generate_frames():
    while True:
        success, frame = camera.read()
        classIds, confs, bbox = net.detect(frame,confThreshold=0.5)
        #print(classIds)
        if not success:
            break
        else:
            if len(classIds)!= 0:
                for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                    cv2.rectangle(frame,box, color=(0,255,0), thickness=2)
                    cv2.putText(frame,classNames[classId-1],(box[0]+10, box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),2)
                    cv2.putText(frame,str(round(confidence*100.2)),(box[0]+200, box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),2)
                engine=pyttsx3.init()
                engine.say("Awas Ada")
                engine.say(classNames[classId-1])
                engine.runAndWait()
                #time.sleep(1)
                
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpg\r\n\r\n' + frame + b'\r\n')
            
    
    key = cv2.waitKey(0)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/deskripsi')
def deskripsi():
    return render_template('deskripsi.html')
@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/video_feed')

def video_feed():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')
    

if __name__=="__main__":
    app.run()