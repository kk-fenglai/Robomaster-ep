import cv2  
import numpy as np  
import time  
from robomaster import robot  
from robomaster import camera  
  
def find_center_point(frame):  
    # 假设frame是一个NumPy数组，代表图像  
    height, width = frame.shape[:2]  
    center_x = width // 2  
    center_y = height // 2  
    return center_x, center_y  
  
if __name__ == '__main__':  
    ep_robot = robot.Robot()  
    ep_robot.initialize(conn_type="ap")  
  
    ep_camera = ep_robot.camera  
  
    # 开始视频流，但不显示，以便我们可以处理帧  
    ep_camera.start_video_stream(display=True, resolution=camera.STREAM_360P)  

  
    try:  
        while True:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)

            center_x, center_y = find_center_point(frame)
            print(f"Center point: ({center_x}, {center_y})")
                # 这里您可以根据需要处理frame，例如显示、保存或进行其他分析  
            else:  
                break  
            time.sleep(0.1)  # 稍微暂停一下，避免处理过快  
  
            # 如果您只需要处理一定数量的帧，可以添加一个计数器并在达到某个值后退出循环  
    finally:  
        # 确保在结束时关闭VideoCapture和视频流  
        ep_camera.stop_video_stream()  
        cap.release()  
  
    ep_robot.close()