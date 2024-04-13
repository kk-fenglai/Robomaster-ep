import cv2
import numpy as np

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取视频流中的一帧
    ret, frame = cap.read()
    if not ret:
        break

    # 将帧转换为HSV颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 定义黑色的HSV范围
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([179, 255, 30])

    # 创建黑色的掩膜
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # 执行形态学操作以去除噪声
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 找到轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制轮廓
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('Detected Black Lines', frame)

    # 检测键盘按键，如果按下'q'键则退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()