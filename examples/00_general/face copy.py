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
import threading  
  






#获取官方提供的特征库，根据自己电脑设置路径
face_cascade = cv2.CascadeClassifier("D:/gooledownloads/anaconda/envs/classic/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")  
eye_cascade = cv2.CascadeClassifier("D:/gooledownloads/anaconda/envs/classic/Lib/site-packages/cv2/data/haarcascade_eye.xml")
KP = 0.15#比例系数，让云台转的慢一点


  
def detect_face_and_eyes(img):  
    # 这里可以进行一些优化，比如降低图像分辨率  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    gray = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)  
      
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)  
    face_center = None  
    if len(faces) > 0:  
        for faceRect in faces:  
            x, y, w, h = [v * 2 for v in faceRect]  # 因为降低了分辨率，所以需要乘以2  
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)  
            roi_gray = gray[y:y + h // 2, x:x + w]  
            roi_color = img[y:y + h // 2, x:x + w]  
            eyes = eye_cascade.detectMultiScale(roi_gray)  
            for (ex, ey, ew, eh) in eyes:  
                ex, ey, ew, eh = [v * 2 for v in (ex, ey, ew, eh)]  # 同样需要乘以2  
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)  
            face_center = (x + w // 2, y + h // 2)  
    return img, face_center  
  
def control_gimbal(face_center, image_center):  
    if face_center is None:  
        return  
    yaw0_speed = 0  
    pitch0_speed = 0  
    error_x = image_center[0] - face_center[0]  
    error_y = image_center[1] - face_center[1]  
    if abs(error_x) > 10:  
        yaw0_speed = KP * error_x  
    if abs(error_y) > 10:  
        pitch0_speed = -KP * error_y  
    # 在这里，你可以调用控制云台的函数，但确保它不会阻塞主线程  
    # ep_gimbal.drive_speed(pitch_speed=pitch0_speed, yaw_speed=yaw0_speed)  



def detect_and_control(img):  
    img, face_center = detect_face_and_eyes(img)  
    control_gimbal(face_center, image_center)  
    cv2.imshow("img", img)  
# 主循环  
if __name__ == '__main__':

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


    ep_chassis = ep_robot.chassis
    image_center = (400, 200)  # 设定图像中心坐标  
    try:
        while True:  
            img = ep_camera.read_cv2_image()  
            
            # 使用线程进行人脸检测，以避免阻塞主线程  
            
            # 创建一个线程来执行检测和控制任务  
            t = threading.Thread(target=detect_and_control, args=(img,))    
            t.start()  
    except KeyboardInterrupt:  
        # 当用户按下Ctrl+C时退出循环  
        print("Program is shutting down...")  
    except Exception as e:  
        # 处理其他异常  
        print(f"An error occurred: {e}")  
    finally:  
        # 在循环外取消订阅TOF数据并关闭机器人连接  
        cv2.destroyAllWindows()
        ep_robot.close()


    

#     import cv2  
# import threading  
  
# # 假设 face_cascade 和 eye_cascade 已经被正确加载  
  
# def detect_face_and_eyes(img):  
#     # 这里可以进行一些优化，比如降低图像分辨率  
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
#     gray = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)  
      
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)  
#     face_center = None  
#     if len(faces) > 0:  
#         for faceRect in faces:  
#             x, y, w, h = [v * 2 for v in faceRect]  # 因为降低了分辨率，所以需要乘以2  
#             cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)  
#             roi_gray = gray[y:y + h // 2, x:x + w]  
#             roi_color = img[y:y + h // 2, x:x + w]  
#             eyes = eye_cascade.detectMultiScale(roi_gray)  
#             for (ex, ey, ew, eh) in eyes:  
#                 ex, ey, ew, eh = [v * 2 for v in (ex, ey, ew, eh)]  # 同样需要乘以2  
#                 cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)  
#             face_center = (x + w // 2, y + h // 2)  
#     return img, face_center  
  
# def control_gimbal(face_center, image_center):  
#     if face_center is None:  
#         return  
#     yaw0_speed = 0  
#     pitch0_speed = 0  
#     error_x = image_center[0] - face_center[0]  
#     error_y = image_center[1] - face_center[1]  
#     if abs(error_x) > 10:  
#         yaw0_speed = KP * error_x  
#     if abs(error_y) > 10:  
#         pitch0_speed = -KP * error_y  
#     # 在这里，你可以调用控制云台的函数，但确保它不会阻塞主线程  
#     # ep_gimbal.drive_speed(pitch_speed=pitch0_speed, yaw_speed=yaw0_speed)  
  
# # 主循环  
# def main_loop():  
#     # ... 初始化代码 ...  
      
#     while True:  
#         img = ep_camera.read_cv2_image()  
          
#         # 使用线程进行人脸检测，以避免阻塞主线程  
#         def detect_and_control():  
#             img, face_center = detect_face_and_eyes(img)  
#             control_gimbal(face_center, image_center)  
#             cv2.imshow("img", img)  
          
#         # 创建一个线程来执行检测和控制任务  
#         t = threading.Thread(target=detect_and_control)  
#         t.start()  
          
#         # 主线程可以继续进行其他

    
    

