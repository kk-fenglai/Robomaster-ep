import mediapipe as mp
import cv2

# 初始化 MediaPipe 的人体关键点模型
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 初始化 PID 控制器参数
kp = 0.1  # 比例参数
ki = 0.01  # 积分参数
kd = 0.01  # 微分参数
prev_error = 0  # 上一时刻误差
integral = 0  # 误差累积

# 设置机器人移动速度
robot_speed = 0.1

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 初始化 MediaPipe 人体姿势模型
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        # 将图像转换为 RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 检测人体姿势
        results = pose.process(image_rgb)

        if results.pose_landmarks is not None:
            # 获取关键点坐标
            landmarks = results.pose_landmarks.landmark

            # 计算人体中心坐标
            center_x = sum([landmark.x for landmark in landmarks]) / len(landmarks)
            center_y = sum([landmark.y for landmark in landmarks]) / len(landmarks)

            # 计算中心与画面中心的偏差
            error = center_x - 0.5  # 假设画面宽度为 1，中心为 0.5

            # 计算 PID 控制器的输出
            integral += error
            derivative = error - prev_error
            output = kp * error + ki * integral + kd * derivative

            # 更新上一时刻误差
            prev_error = error

            # 根据 PID 控制器输出调整机器人的移动方向
            # 这里简单地假设机器人是通过速度来控制移动方向的，你可能需要根据实际情况进行调整
            # 这里的代码是一个简单示例，实际情况可能会更加复杂
            # 你需要根据具体情况来控制机器人的移动，例如调整电机速度、舵机角度等
            # 这里仅仅是一个示例，需要根据具体情况进行修改
            if output > 0:
                # 向右转
                # 在这里你可以添加具体的机器人控制代码
                print("Turn right")
            else:
                # 向左转
                # 在这里你可以添加具体的机器人控制代码
                print("Turn left")

        # 在图像上绘制人体骨架
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 显示结果图像
        cv2.imshow('MediaPipe Pose', frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()
