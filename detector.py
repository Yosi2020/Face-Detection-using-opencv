import cv2
import numpy as np
import os
import sqlite3

faceDetect = cv2.CascadeClassifier(r'D:\Python\opencv\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

rec = cv2.face.LBPHFaceRecognizer_create();
rec.read('recognizer\\trainningData.yml')
font = cv2.FONT_HERSHEY_SIMPLEX

def getProfile(id):
    conn = sqlite3.connect("FaceBase.db")
    cmd = "SELECT * FROM Data WHERE ID = " + str(id)
    cursor = conn.execute(cmd)
    profile = None
    for row in cursor:
        profile = row
    conn. close()
    return profile

while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.05, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,255),3)
        id, conf = rec.predict(gray[y:y+h, x:x+w])
        profile = getProfile(id)
         # If confidence is less them 100 ==> "0" : perfect match 
        if (conf < 100):
            id = profile[id]
            conf = "  {0}%".format(round(100 - conf))
            cv2.putText(img, 'Name:- ' + str(profile[1]),(x,y+h+30),font,1,(192,192,192),2)
            cv2.putText(img, "Age:- " + str(profile[2]),(x,y+h+60),font,1,(0,0,51),2)
            cv2.putText(img, "Depart:-" + str(profile[3]),(x,y+h+90),font,1,(0,204,204),2)
            cv2.putText(img, "Gender:- " + str(profile[4]),(x,y+h+120),font,1,(153,66,0),2);    

        else:
            id = "unknown"
            cv2.putText(img, 'Name:- ' + str(id),(x,y+h+30),font,1,(192,192,192),2);
            cv2.putText(img, "Age:- " + str(id),(x,y+h+60),font,1,(0,0,51),2);
            cv2.putText(img, "Depart:-" + str(id),(x,y+h+90),font,1,(0,204,204),2);
            cv2.putText(img, "Gender:- " + str(id),(x,y+h+120),font,1,(153,66,0),2);
            conf = "  {0}%".format(round(100 - conf))
        cv2.putText(img, "Perfect_Match:-" + str(conf),(x,y+h+150),font,1,(0,0,0),2);
    cv2.imshow('face',img)
    if(cv2.waitKey(1) == ord('s')):
        break
cam.release()
cv2.destroyAllWindows()
