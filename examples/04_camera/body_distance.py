import cv2
import mediapipe as mp
import math
import time

# 初始化MediaPipe姿态检测模块
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 摄像头参数（这些值需要根据实际摄像头进行校准）
# focal_length和pixel_size的值需要根据你的摄像头进行设定
focal_length = 800  # 焦距，单位通常是像素
pixel_size = 500  # 像素大小，单位通常是米/像素

# 函数：根据检测到的关键点计算人体在图像中的像素大小
def calculate_pixel_size(landmarks):
    # 选择关键点来计算人体大小，这里使用左肩和右肩
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    # 确保我们得到了有效的坐标
    if left_shoulder.visibility and right_shoulder.visibility:
        # 计算两点之间的欧几里得距离
        pixel_distance = math.sqrt((right_shoulder.x - left_shoulder.x) ** 2 + (right_shoulder.y - left_shoulder.y) ** 2)
        return pixel_distance
    else:
        return None

# 函数：根据像素大小和摄像头参数估计距离
def estimate_distance(focal_length, pixel_size, pixel_distance, assumed_body_width):
    # 使用相似三角形原理计算距离
    # 实际物体大小（这里假设为人体肩宽）需要是已知的，这里为了演示，我们假设它为0.5米
    estimated_distance = (focal_length * assumed_body_width) / (pixel_size * pixel_distance)
    return estimated_distance

# 打开摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

# 假设的人体肩宽，单位：米
assumed_body_width = 0.5

# （2）处理每一帧图像
lmlist = []  # 存放人体关键点信息

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 将图像从BGR转换为RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 进行姿态检测
    results = pose.process(rgb_frame)

    # 保存检测到的人数及其关键点信息
    person_info = []

    # 绘制姿态关键点
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmark_list = results.pose_landmarks.landmark
        if landmark_list:
            # 计算每个人体的距离并保存
            for idx, landmark in enumerate(landmark_list):
                # 计算人体在图像中的像素大小
                pixel_size_in_pixel = calculate_pixel_size(results.pose_landmarks)

                # 如果成功计算了像素大小，则估计距离
                if pixel_size_in_pixel is not None:
                    estimated_distance = estimate_distance(focal_length, pixel_size, pixel_size_in_pixel, assumed_body_width)
                    print(f"Person {idx+1}: Estimated distance from camera: {estimated_distance:.2f} meters")

                    # 将人体距离和索引保存到列表中
                    person_info.append((estimated_distance, idx))

    # 选择最近的人作为参考目标
    if person_info:
        closest_person = min(person_info)
        print(f"Closest person: Person {closest_person[1]+1} (Distance: {closest_person[0]:.2f} meters)")
    else:
        print("No person detected.")

    # results = pose_estimation_function(frame)
    if results is not None and results.pose_landmarks is not None:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
        # 保存每帧图像的宽、高、通道数
            h, w, c = frame.shape

            # 得到的关键点坐标x/y/z/visibility都是比例坐标，在[0,1]之间
            # 转换为像素坐标(cx,cy)，图像的实际长宽乘以比例，像素坐标一定是整数
            cx, cy = int(landmark.x * w), int(landmark.y * h)

            # 打印坐标信息
            # print(f"Person {idx+1}, Point {mp_pose.PoseLandmark(idx).name}: ({cx}, {cy})")

            # 保存坐标信息
            lmlist.append((cx, cy))

            # 在关键点上画圆圈，img画板，以(cx,cy)为圆心，半径5，颜色绿色，填充圆圈
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
    else:
        print("姿态估计失败或没有找到任何姿态地标。")


    cv2.imshow('Real-time Pose Detection and Distance Estimation', frame)

    # 等待按键，如果按下'q'键则退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源并关闭窗口
cap.release()
cv2.destroyAllWindows()
