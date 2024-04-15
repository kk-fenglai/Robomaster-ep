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


def on_detect_person(person_info):
    number = len(person_info)
    persons.clear()
    for i in range(0, number):
        x, y, w, h = person_info[i]
        persons.append(PersonInfo(x, y, w, h))
        print("person: x:{0}, y:{1}, w:{2}, h:{3}".format(x, y, w, h))
        ep_chassis.move(x=x, y=0, z=math.atan((5-x*10)/5), xy_speed=0.5)


def sub_data_handler(sub_info):
    distance = sub_info
    print("tof1:{0}  tof2:{1}  tof3:{2}  tof4:{3}".format(distance[0], distance[1], distance[2], distance[3]))
    # if distance[0] > 100 or distance[1] > 100:
    #     if distance[0] < 200:
    #         distance[0] = 0
    #     elif distance[1] < 200:
    #         distance[1] = 0
    #     ep_chassis.drive_wheels(w1=distance[0]/80,w2=distance[1]/80,w3=distance[0]/80,w4=distance[1]/80)
    #     time.sleep(1)


if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_chassis=ep_robot.chassis
    ep_sensor = ep_robot.sensor
    ep_vision = ep_robot.vision
    ep_camera = ep_robot.camera

    ep_camera.start_video_stream(False)

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


