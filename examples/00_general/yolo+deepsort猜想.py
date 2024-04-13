import cv2  
from robomaster import robot  # 假设你已经安装了RoboMaster SDK  
from yolo import YOLO  # YOLO检测器  
from deepsort import DeepSORT  # DeepSORT跟踪器  
  
# 初始化RoboMaster机器人  
ep_robot = robot.Robot()  
ep_robot.initialize()  
  
# 初始化YOLO和DeepSORT  
yolo = YOLO()  
deepsort = DeepSORT()  
  
# 读取视频流（这里假设使用机器人自带的摄像头）  
camera = ep_robot.camera  
  
while True:  
    # 从摄像头获取图像帧  
    frame = camera.capture_frame()  
      
    # 运行YOLO检测  
    boxes, scores, classes = yolo.detect(frame)  
      
    # 提取特征并运行DeepSORT跟踪  
    features = extract_features(frame, boxes)  
    tracks = deepsort.update(boxes, scores, classes, features)  
      
    # 根据跟踪结果计算控制指令  
    control_commands = calculate_control_commands(tracks, ep_robot.position)  
      
    # 发送控制指令给机器人  
    for command in control_commands:  
        ep_robot.move(command['direction'], command['speed'])  
        # 可以添加其他控制指令，如射击等  
      
    # 显示处理后的帧（可选，用于调试）  
    cv2.imshow('Tracking', frame)  
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  
  
# 关闭机器人连接和窗口  
ep_robot.close()  
cv2.destroyAllWindows()