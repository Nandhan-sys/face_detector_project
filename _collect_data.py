import cv2
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
data=[]
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        face_present=1
        face_size=w*h
        #face_centered  1 or 0
        face_centered=1 if abs((x+w/2)-frame.shape[1]/2)<50 else 0
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        key=cv2.waitKey(1) & 0xFF
        label=None
        if key==ord('f'):
            label='focused'
        elif key==ord('d'):
            label='distracted'
        if label is not None:
              data.append([face_present,face_size,face_centered,label])
              print("Saved",label)
    cv2.imshow('frame',frame)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
import pandas as pd
df=pd.DataFrame(data,columns=['face_present','face_size','face_centered','label'])
df.to_csv('driver_attention_data.csv',index=False)
df.head()
import os
print(os.getcwd())

