# # Coding BIGBOSSyifi
# # Datatime:2022/4/24 21:41
# # Filename:HandsDetector.py
# # Toolby: PyCharm
#
# import cv2
# import mediapipe as mp
# import time
#
# cap = cv2.VideoCapture(0)  # OpenCV摄像头调用：0=内置摄像头（笔记本）   1=USB摄像头-1  2=USB摄像头-2
#
# # 定义并引用mediapipe中的hands模块
# mpHands = mp.solutions.hands
# hands = mpHands.Hands()
# mpDraw = mp.solutions.drawing_utils
#
# # 帧率时间计算
# pTime = 0
# cTime = 0
#
# while True:
#     success, img = cap.read()
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2图像初始化
#     results = hands.process(imgRGB)
#     # print(results.multi_hand_landmarks)
#
#     if results.multi_hand_landmarks:
#         for handLms in results.multi_hand_landmarks:
#             for id, lm in enumerate(handLms.landmark):
#                 # print(id, lm)
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 print(id, cx, cy)
#                 # if id == 4:
#                 cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
#
#             # 绘制手部特征点：
#             mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
#     '''''
#   视频FPS计算
#      '''
#     cTime = time.time()
#     fps = 1 / (cTime - pTime)
#     pTime = cTime
#
#     cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
#                 (255, 0, 255), 3)  # FPS的字号，颜色等设置
#
#     cv2.imshow("HandsImage", img)  # CV2窗体
#     cv2.waitKey(1)  # 关闭窗体
import cv2
import mediapipe as mp

# mp.solutions.drawing_utils用于绘制
mp_drawing = mp.solutions.drawing_utils

# 参数：1、颜色，2、线条粗细，3、点的半径
DrawingSpec_point = mp_drawing.DrawingSpec((0, 255, 0), 1, 1)
DrawingSpec_line = mp_drawing.DrawingSpec((0, 0, 255), 1, 1)

# mp.solutions.holistic是一个类别，是人的整体
mp_holistic = mp.solutions.holistic

# 参数：1、是否检测静态图片，2、姿态模型的复杂度，3、结果看起来平滑（用于video有效），4、检测阈值，5、跟踪阈值
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
while cap.isOpened():
  success, image = cap.read()
  if not success:
    print("Ignoring empty camera frame.")
    continue
  image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # 处理RGB图像
  results = holistic.process(image1)

  '''
  mp_holistic.PoseLandmark类中共33个人体骨骼点
  mp_holistic.HandLandmark类中共21个手部关键点
  脸部有468个关键点
  '''

  # 绘制
  mp_drawing.draw_landmarks(
    image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, DrawingSpec_point, DrawingSpec_line)
  mp_drawing.draw_landmarks(
    image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, DrawingSpec_point, DrawingSpec_line)
  mp_drawing.draw_landmarks(
    image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, DrawingSpec_point, DrawingSpec_line)
  mp_drawing.draw_landmarks(
    image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, DrawingSpec_point, DrawingSpec_line)

  cv2.imshow('MediaPipe Holistic', image)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

holistic.close()
cv2.destroyAllWindows()
cap.release()