import robomaster  
from robomaster import robot  
import time  
  
# 全局变量定义  
test = 0  
move_in_progress = False  # 用于标识是否已有移动命令在执行  
count=0
def sub_data_handler(distance):  
    global test, move_in_progress ,count
    # 根据第一个TOF传感器的距离值来更新test变量  
    print("tof1:{0}  tof2:{1}  tof3:{2}  tof4:{3}".format(distance[0], distance[1], distance[2], distance[3])) 
    if distance[0] < 500:  
        if not move_in_progress:  # 如果没有移动在进行，则开始移动  
            move_robot()  
    else:
        if count<=2:
            ep_chassis.drive_speed(x=0.5, y=0, z=0)
        else:
            ep_chassis.move(x=1.2, y=0, z=0, xy_speed=0.7).wait_for_completed()
            ep_chassis.stop()
        move_in_progress = False  # 停止时重置移动标识  
  
def move_robot():
    global move_in_progress,count
    move_in_progress = True  # 设置移动标识为True  
    try:  
        # 左移 0.6米  
        ep_chassis.move(x=0, y=-0.6, z=0, xy_speed=0.7).wait_for_completed()  
        # 前移 0.5米  
        ep_chassis.move(x=1.2, y=0, z=0, xy_speed=0.7).wait_for_completed()  
        # 右移 0.6米  
        ep_chassis.move(x=0, y=0.6, z=0, xy_speed=0.7).wait_for_completed()
        count+=1
    except Exception as e:  
        print(f"An error occurred while moving the robot: {e}")  
    finally:  
        move_in_progress = False  # 移动完成后重置移动标识  
  
if __name__ == '__main__':  
    # ep_robot = robot.Robot()
    ep_robot = robot.Robot()  
    ep_robot.initialize(conn_type="ap")  
    ep_sensor = ep_robot.sensor
    ep_chassis = ep_robot.chassis 
    ep_sensor.sub_distance(freq=5, callback=sub_data_handler)  
    try:  
        while True:  
            # 主循环可以保持空闲，或者执行其他任务  
            time.sleep(0.1)  
    except KeyboardInterrupt:  
        # 当用户按下Ctrl+C时退出循环  
        print("Program is shutting down...")  
    except Exception as e:  
        # 处理其他异常  
        print(f"An error occurred: {e}")  
    finally:  
        # 在循环外取消订阅TOF数据并关闭机器人连接  
        ep_sensor.unsub_distance()   
        ep_robot.close()