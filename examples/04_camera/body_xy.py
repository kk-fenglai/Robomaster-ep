import cv2
import mediapipe as mp
import time

# 导入姿态跟踪方法
mpPose = mp.solutions.pose  # 姿态识别方法
pose = mpPose.Pose(static_image_mode=False,  # 静态图模式，False代表置信度高时继续跟踪，True代表实时跟踪检测新的结果
                   # upper_body_only=False,  # 是否只检测上半身
                   smooth_landmarks=True,  # 平滑，一般为True
                   min_detection_confidence=0.5,  # 检测置信度
                   min_tracking_confidence=0.5)  # 跟踪置信度
# 检测置信度大于0.5代表检测到了，若此时跟踪置信度大于0.5就继续跟踪，小于就沿用上一次，避免一次又一次重复使用模型

# 导入绘图方法
mpDraw = mp.solutions.drawing_utils

# （1）导入视频
cap = cv2.VideoCapture(0)

pTime = 0  # 设置第一帧开始处理的起始时间

# （2）处理每一帧图像
lmlist = []  # 存放人体关键点信息

def get_landmark_pos(landmark, width, height):
    return [landmark.x * width, landmark.y * height]

while True:

    # 接收图片是否导入成功、帧图像
    success, img = cap.read()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 将导入的BGR格式图像转为RGB格式
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 将图像传给姿态识别模型
    results = pose.process(imgRGB)

    # 查看体态关键点坐标，返回x,y,z,visibility
    # print(results.pose_landmarks)

    # 如果检测到体态就执行下面内容，没检测到就不执行
    if results.pose_landmarks:

        # 绘制姿态坐标点，img为画板，传入姿态点坐标，坐标连线
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        # 获取32个人体关键点坐标, index记录是第几个关键点
        for index, lm in enumerate(results.pose_landmarks.landmark):
            # 保存每帧图像的宽、高、通道数
            h, w, c = img.shape

            # 得到的关键点坐标x/y/z/visibility都是比例坐标，在[0,1]之间
            # 转换为像素坐标(cx,cy)，图像的实际长宽乘以比例，像素坐标一定是整数
            cx, cy = int(lm.x * w), int(lm.y * h)

            # 打印坐标信息
            # print(index, cx, cy)
            print(h, w, c)

            # 保存坐标信息
            lmlist.append((cx, cy))
            # print(lmlist)

            # 在关键点上画圆圈，img画板，以(cx,cy)为圆心，半径5，颜色绿色，填充圆圈
            cv2.circle(img, (cx, cy), 3, (0, 255, 0), cv2.FILLED)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # 左臂坐标
            left_shoulder = get_landmark_pos(
                landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value], width, height)
            left_elbow = get_landmark_pos(
                landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value], width, height)
            left_wrist = get_landmark_pos(
                landmarks[mpPose.PoseLandmark.LEFT_WRIST.value], width, height)
            left_hip = get_landmark_pos(
                landmarks[mpPose.PoseLandmark.LEFT_HIP.value], width, height)
            left_knee = get_landmark_pos(
                landmarks[mpPose.PoseLandmark.LEFT_KNEE.value], width, height)
            left_eye = get_landmark_pos(
                landmarks[mpPose.PoseLandmark.LEFT_EYE_OUTER.value], width, height)

            # 右臂坐标
            right_shoulder = get_landmark_pos(
                landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value], width, height)
            right_elbow = get_landmark_pos(
                landmarks[mpPose.PoseLandmark.RIGHT_ELBOW.value], width, height)
            right_wrist = get_landmark_pos(
                landmarks[mpPose.PoseLandmark.RIGHT_WRIST.value], width, height)
            right_hip = get_landmark_pos(
                landmarks[mpPose.PoseLandmark.RIGHT_HIP.value], width, height)
            right_knee = get_landmark_pos(
                landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value], width, height)
            right_eye = get_landmark_pos(
                landmarks[mpPose.PoseLandmark.RIGHT_EYE_OUTER.value], width, height)
            nose = get_landmark_pos(
                landmarks[mpPose.PoseLandmark.NOSE.value], width, height)


            # 计算矩形的中心点坐标
            center_x = (left_shoulder[0] + right_shoulder[0] ) / 2
            center_y = (left_shoulder[1] + right_shoulder[1] ) / 2

            # # 定义目标位置（画面中心）和PID控制器的参数
            target_x = w / 2  # 画面中心的x坐标
            target_y = h / 2  # 画面中心的y坐标
            print(center_x, center_y,target_x,target_y)


    # 查看FPS
    cTime = time.time()  # 处理完一帧图像的时间
    fps = 1 / (cTime - pTime)
    pTime = cTime  # 重置起始时间

    # 在视频上显示fps信息，先转换成整数再变成字符串形式，文本显示坐标，文本字体，文本大小
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 显示图像，输入窗口名及图像数据
    cv2.imshow('image', img)
    if cv2.waitKey(10) & 0xFF == 27:  # 每帧滞留15毫秒后消失，ESC键退出
        break

# 释放视频资源
cap.release()
cv2.destroyAllWindows()