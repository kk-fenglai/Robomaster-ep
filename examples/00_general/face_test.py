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
  
#创建新的对象
ep_robot = robot.Robot()

# 指定连接方式为AP 直连模式，初始化
ep_robot.initialize(conn_type='ap')   
ep_camera = ep_robot.camera 
ep_gimbal = ep_robot.gimbal

#开始获取视频流，但是不播放
ep_camera.start_video_stream(display=False)

#设置模式为自由模式
ep_robot.set_robot_mode(mode=robot.FREE)





#获取官方提供的特征库，根据自己电脑设置路径
face_cascade = cv2.CascadeClassifier("D://python38//Lib//site-packages//cv2//data//haarcascade_frontalface_default.xml")  
eye_cascade = cv2.CascadeClassifier("D://python38//Lib//site-packages//cv2//data//haarcascade_eye.xml")
KP = 0.15#比例系数，让云台转的慢一点


  
def detect_face_and_eyes(img):  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)  
    face_center = None  
    if len(faces) > 0:  
        for faceRect in faces:  
            x, y, w, h = faceRect  
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)  
            roi_gray = gray[y:y + h // 2, x:x + w]  
            roi_color = img[y:y + h // 2, x:x + w]  
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 1, cv2.CASCADE_SCALE_IMAGE, (2, 2))  
            for (ex, ey, ew, eh) in eyes:  
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)  
            face_center = (x + w // 2, y + h // 2)  # 计算人脸中心坐标  
    return img, face_center  
  
def control_gimbal(face_center, image_center):  
    if face_center is None:  
        return  # 如果没有检测到人脸，则不控制云台  
    yaw0_speed = 0  
    pitch0_speed = 0  
    error_x = image_center[0] - face_center[0]  
    error_y = image_center[1] - face_center[1]  
    if abs(error_x) > 10:  
        yaw0_speed = KP * error_x  
    if abs(error_y) > 10:  
        pitch0_speed = -KP * error_y  
    print("Yaw speed:", yaw0_speed)  
    print("Pitch speed:", pitch0_speed)  
    ep_gimbal.drive_speed(pitch_speed=pitch0_speed, yaw_speed=yaw0_speed)  
  
# 主循环  
image_center = (400, 200)  # 设定图像中心坐标  
while True:  
    img = ep_camera.read_cv2_image()  
    img, face_center = detect_face_and_eyes(img)  
    cv2.imshow("img", img)  
    if face_center is not None:  
        control_gimbal(face_center, image_center)  
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按'q'键退出循环  
        break  
  
cv2.destroyAllWindows()
ep_robot.close()