import cv2 as cv
import numpy as np
import time


face = cv.CascadeClassifier("Resources/haar_face.xml")

vid =cv.VideoCapture(0)

vid.set (3,640)
vid.set (4,480)

center=(0,0)
score=0
prev = None

while True:
    success,img=vid.read()
    flip=cv.flip(img,+1)
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    faces=face.detectMultiScale(gray,1.1,5)

    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        center=((x+w//2),(y+h)//2)

    # cv.putText(img,f'{score}',(20,20),cv.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)

    if prev is not None:
        pass
    
    #yo le chai yedi right tilt vayo vane wala score ho ignore it
    # if center[0]>flip.shape[1]//2 :
    #     time.sleep(1)i
    #     score=score+1
    prev = gray
    cv.putText(flip, f"Score: {score}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv.imshow("Live",flip)
    if cv.waitKey(10) & 0xFF==ord("f"):
        break
    


cv.waitKey(1)
cv.destroyAllWindows()