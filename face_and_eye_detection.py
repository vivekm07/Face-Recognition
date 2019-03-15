import numpy as np
import cv2    #import OpenCV library

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')                   
right_eye = cv2.CascadeClassifier('haarcascade_righteye.xml')                #reading haarcascade XML files
left_eye = cv2.CascadeClassifier('haarcascade_lefteye.xml')

cap = cv2.VideoCapture(0)               #function for taking input_video_stream

while 1:
    ret, img = cap.read()      
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     #function to convert image in grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray_face = gray[y:y+h, x:x+w]                      #loop for detecting face
        roi_color_face = img[y:y+h, x:x+w]
        cv2.imwrite('full.jpg',img)
        
        r_eye = right_eye.detectMultiScale(roi_gray_face)
        l_eye = left_eye.detectMultiScale(roi_gray_face)            #writing detection result in respective variables
        cv2.imwrite('img_face.jpg',roi_color_face)

        for (rx,ry,rw,rh) in r_eye:
            r_pic = roi_color_face[ry:ry+rh,rx:rx+rw]
            cv2.rectangle(roi_color_face,(rx,ry),(rx+rw,ry+rh),(0,255,0),2)        #loop for detecting right eye
            cv2.imwrite('eye_right.jpg',r_pic)

        for (lx,ly,lw,lh) in l_eye:
            l_pic = roi_color_face[ly:ly+lh,lx:lx+lw]                               
            cv2.rectangle(roi_color_face,(lx,ly),(lx+lw,ly+lh),(0,255,0),2)        #loop for detecting left eye
            cv2.imwrite('eye_left.jpg',l_pic)
        

    cv2.imshow('img',img)                           #function for displaying image 
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
