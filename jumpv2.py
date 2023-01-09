import cv2 as cv
import numpy as np


vid=cv.VideoCapture(0)
framerate=30
minjtime=1
threshjump=framerate*minjtime

center=None
prev=0
score=0

haar=cv.CascadeClassifier("Resources/haar_face.xml")

while True:
    success,frame=vid.read()
    

    flip=cv.flip(frame,+1)
    gray=cv.cvtColor(flip,cv.COLOR_BGR2GRAY)
    blur=cv.medianBlur(gray,3)

    faces=haar.detectMultiScale(blur,1.1,5)
    # for (x,y,w,h) in faces:
    #     rect=cv.rectangle(flip,(x,y),((x+w),(y+h)),(0,255,0),2)
    if len(faces)>0:
        (x,y,w,h)=faces[0]
        center=(x+w//2,y+w//2)    
    if center is not None and len(faces)==0:
        prev= prev+1
        if prev>threshjump:
            score=score+1
            prev=0
        center=None

    if center is not None:
        cv.circle(flip,center,5,(0,255,0))

    cv.putText(flip,f'your score is {score}',(20,20),cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    cv.imshow("LIVE CAM",flip)
    if cv.waitKey(10) & 0xFF==ord("f"):
        break


cv.waitKey(1)
cv.destroyAllWindows()