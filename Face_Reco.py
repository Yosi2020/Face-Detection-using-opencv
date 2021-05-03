import cv2 as cv
import numpy as np
import sqlite3

faceDetect= cv.CascadeClassifier('C:/Users/deribe/Desktop/Python/Important file/opencv/opencv-master/data/haarcascades_cuda/haarcascade_frontalface_default.xml')
cam = cv.VideoCapture(0)

def insertOrUpdate(Id,Name):
    conn = sqlite3.connect("Facebase.db")
    cmd = "SELECT * FROM data WHERE id ="+ str(Id)
    cursor = conn.execute(cmd)
    isRecordExist = 0
    for row in cursor:
        isRecordExist = 1
    if (isRecordExist == 1):
        cmd = "UPDATE data SET Name =" + str(Name)+ " WHERE id =" + str(Id)
    else:
        cmd = "INSERT INTO data(id,Name) Values("+str(Id)+","+str(Name)+")"
    conn.execute(cmd)
    conn.commit()
    conn.close()

id = input('Enter your id:- ')
name = input('Enter your name:- ')
name = name.title()  # or name.Capitialize()
name = name.strip()
insertOrUpdate(id,name)
sampleNum = 0
while(True):
    ret, img = cam.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.05, 4)
    for (x,y,w,h) in faces:
        sampleNum = sampleNum + 1
        cv.imwrite('Data creater/User.'+str(id)+ '.'+ str(sampleNum)+'.jpg',gray[y:y+h,x:x+w])
        cv.rectangle(img, (x,y), (x+w, y+h), (0,0,255),5)
        cv.waitKey(200)
    cv.imshow('Face', img)    
    if(sampleNum > 30):
        break
    
cam.release()
cv.destroyAllWindows()
