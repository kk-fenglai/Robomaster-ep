#导入需要用到的库
import cv2
import numpy as np
import wave
import re
import socket
import sys
import numpy as np 
import robomaster
from robomaster import robot
  
import cv2

face_cascade = cv2.CascadeClassifier("C:/software/anaconda3/envs/robomaster/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")  
eye_cascade = cv2.CascadeClassifier("C:/software/anaconda3/envs/robomaster/Lib/site-packages/cv2/data/haarcascade_eye.xml")

def detect_face_and_eyes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
    return image

def main():
    cv2.namedWindow("Face and Eyes Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face and Eyes Detection", 800, 400)

    cap = cv2.VideoCapture(0)  # 使用第一个摄像头，如果有多个摄像头可以选择其他索引或者视频文件路径
    
    while True:
        ret, frame = cap.read()  # 读取视频帧
        if not ret:
            break
        
        processed_frame = detect_face_and_eyes(frame)
        cv2.imshow("Face and Eyes Detection", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


    

    

    
    

