# -*-coding:utf-8-*-
# Copyright (c) 2020 DJI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import cv2
import robomaster
from robomaster import robot
from robomaster import vision
import time
import math
from robomaster import camera

class PersonInfo:

    def __init__(self, x, y, w, h):   #x 中心点x轴坐标，y 中心点y轴坐标，w 宽度，h 高度
        self._x = x
        self._y = y
        self._w = w
        self._h = h

    @property   #方法加入@property后，这个方法相当于一个属性，这个属性可以让用户进行使用，而且用户有没办法随意修改。
    def pt1(self):
        return int((self._x - self._w / 2) * 1280), int((self._y - self._h / 2) * 720)

    @property
    def pt2(self):
        return int((self._x + self._w / 2) * 1280), int((self._y + self._h / 2) * 720)

    @property
    def center(self):
        return int(self._x * 1280), int(self._y * 720)


persons = []

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0

    def update(self, current_value):
        error = self.setpoint - current_value
        self.integral += error
        if self.integral > 10:  
            self.integral = 10  
        elif self.integral < -10:  
            self.integral = -10

        derivative = error - self.prev_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

# Constants for PID tuning
KP = 0.1
KI = 0
KD = 0

# Assuming image_center is known
image_center = (640, 480)  # Example image center coordinates



def control_chassis(face_center):
    if face_center is None:
        return  # No face detected, do nothing
    
    error_x = image_center[0] - face_center[0]
    

    # Use PID controller to calculate chassis speed
    chassis_speed = pid_controller.update(error_x)
    if chassis_speed > 2:  
        chassis_speed = 2  
    elif chassis_speed < -2:  
        chassis_speed = -2
    
    # Assuming only x-axis movement for simplicity
    ep_chassis.drive_speed(x=0, y=0, z=chassis_speed )

def on_detect_person(person_info):
    number = len(person_info)
    persons.clear()
    for i in range(0, number):
        x, y, w, h = person_info[i]
        persons.append(PersonInfo(x, y, w, h))
        print("person: x:{0}, y:{1}, w:{2}, h:{3}".format(x, y, w, h))
        face_center = (x + w // 2, y + h // 2)
        control_chassis(face_center)

# Assuming you have a mechanism to detect persons and call on_detect_person function


# def control_gimbal(face_center, image_center):  
#     if face_center is None:  
#         return  # 如果没有检测到人脸，则不控制云台  
#     z_val = 0  
#     pitch0_speed = 0  
#     error_x = image_center[0] - face_center[0]  
#     error_y = image_center[1] - face_center[1]  
#     if abs(error_x) > 10:  
#         z_val = KP * error_x  
#     # if abs(error_y) > 10:  
#     #     pitch0_speed = -KP * error_y  
#     print("z_val:", z_val)  
#     print("Pitch speed:", pitch0_speed)  
#     ep_chassis.drive_speed(x=0, y=0, z=z_val)


# def on_detect_person(person_info):
#     number = len(person_info)
#     persons.clear()
#     for i in range(0, number):
#         x, y, w, h = person_info[i]
#         persons.append(PersonInfo(x, y, w, h))
#         print("person: x:{0}, y:{1}, w:{2}, h:{3}".format(x, y, w, h))
#         ep_chassis.move(x=x, y=0, z=math.atan((5-x*10)/5), xy_speed=0.5)





if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_chassis=ep_robot.chassis
    ep_sensor = ep_robot.sensor
    ep_vision = ep_robot.vision
    ep_camera = ep_robot.camera

    #ep_camera.start_video_stream(False)
    ep_camera.start_video_stream(display=True, resolution=camera.STREAM_360P)
    pid_controller = PIDController(KP, KI, KD, setpoint=0)

    try:
        result = ep_vision.sub_detect_info(name="person", callback=on_detect_person)  #订阅智能识别消息, name=人、手势、线条
        # ep_sensor.sub_distance(freq=5, callback=sub_data_handler)


        while True:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)  # 读取一帧视频流帧
            for j in range(0, len(persons)):
                cv2.rectangle(img, persons[j].pt1, persons[j].pt2, (255, 255, 255))
            cv2.imshow("Persons", img)
            cv2.waitKey(1)  # imshow的持續時間
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        result = ep_vision.unsub_detect_info(name="person")
        cv2.destroyAllWindows()
        ep_camera.stop_video_stream()

        ep_sensor.unsub_distance()
        ep_robot.close()


