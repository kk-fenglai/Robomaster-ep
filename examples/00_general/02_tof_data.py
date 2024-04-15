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


import robomaster
from robomaster import robot
import time
from robomaster import sensor



if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    try:
        ep_sensor = ep_robot.sensor
    # 创建 TofSubject 类的对象
        tof_subject_instance = sensor.TofSubject()

        # 调用对象的方法或访问属性
        distance_info = tof_subject_instance.data_info()
        print(distance_info)  # 打印距离信息
    except KeyboardInterrupt:
        pass
    finally:
        ep_robot.close()
