import cv2  
import torch  
  
# 加载YOLOv5模型  
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5n.pt', force_reload=True)  
model.eval()  
  
# 初始化视频捕获  
cap = cv2.VideoCapture(0)  # 0 通常表示默认摄像头，可以替换为视频文件路径  
  
# 检查是否成功打开摄像头  
if not cap.isOpened():  
    print("Error: Could not open video device or file.")  
    exit()  
  
# 设置分数阈值  
score_threshold = 0.5  
  
while True:  
    # 读取一帧图像  
    ret, frame = cap.read()  
    if not ret:  
        print("Error: Unable to read frame from video stream or file.")  
        break  
  
    # 将图像转换为RGB格式  
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
  
    # 将图像尺寸调整为模型所需的大小（如果需要）  
    # img_rgb = cv2.resize(img_rgb, (width, height))  
  
    # 使用模型进行推理  
    results = model(img_rgb)  
  
    # 处理结果  
    for result in results.xyxy[0]:  
        if result[4] >= score_threshold:  
            box = result[:4]  
            score = result[4]  
            label = int(result[5])  
  
            # 绘制框和标签  
            x1, y1, x2, y2 = map(int, box)  
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  
            cv2.putText(frame, f"{model.names[label]}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  
  
    # 显示结果图像  
    cv2.imshow('YOLOv5 Real-time Object Detection', frame)  
  
    # 按下'q'键退出循环  
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  
  
# 释放资源并关闭窗口  
cap.release()  
cv2.destroyAllWindows()