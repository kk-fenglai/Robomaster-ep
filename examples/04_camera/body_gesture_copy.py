import cv2
import mediapipe as mp
import numpy as np
import scipy.ndimage.filters as filters

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

history_length = 50
angle_history = {
    "left_arm_body_x": np.zeros(history_length),
    "right_arm_body_x": np.zeros(history_length),
    "left_arm_body_y": np.zeros(history_length),
    "right_arm_body_y": np.zeros(history_length),
    "left_arm_forearm": np.zeros(history_length),
    "right_arm_forearm": np.zeros(history_length),
    "head_forearm": np.zeros(history_length),
}
gaussian_filter = filters.gaussian_filter1d

def calculate_center(shoulder_left, shoulder_right, thigh_left, thigh_right):
    # 计算矩形中心点坐标
    center_x = (shoulder_left[0] + shoulder_right[0] + thigh_left[0] + thigh_right[0]) / 4
    center_y = (shoulder_left[1] + shoulder_right[1] + thigh_left[1] + thigh_right[1]) / 4
    return center_x, center_y

def calculate_3d_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def get_landmark_pos(landmark, width, height):
    return [landmark.x * width, landmark.y * height]


def main():
    #视频检测
    #cap = cv2.VideoCapture('你要识别视频的路径')
    #widght = int(cap.get(3))  # 在视频流的帧的宽度,3为编号，不能改
    #height = int(cap.get(4))  # 在视频流的帧的高度,4为编号，不能改
    #size = (width, height)
    #fps = 30  # 帧率
    #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 为视频编码方式，保存为mp4文
    #out = cv2.VideoWriter()
    # 定义一个视频存储对象，以及视频编码方式,帧率，视频大小格式
    #out.open("E:/video.mp4", fourcc, fps, size)
    #摄像头检测
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    with mp_pose.Pose(static_image_mode=False, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # 左臂坐标
                left_shoulder = get_landmark_pos(
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], width, height)
                left_elbow = get_landmark_pos(
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], width, height)
                left_wrist = get_landmark_pos(
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value], width, height)
                left_hip = get_landmark_pos(
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], width, height)
                left_knee = get_landmark_pos(
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value], width, height)
                left_eye = get_landmark_pos(
                    landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value], width, height)

                # 右臂坐标
                right_shoulder = get_landmark_pos(
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], width, height)
                right_elbow = get_landmark_pos(
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value], width, height)
                right_wrist = get_landmark_pos(
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value], width, height)
                right_hip = get_landmark_pos(
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], width, height)
                right_knee = get_landmark_pos(
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value], width, height)
                right_eye = get_landmark_pos(
                    landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value], width, height)
                nose = get_landmark_pos(
                    landmarks[mp_pose.PoseLandmark.NOSE.value], width, height)

                # print(left_shoulder, right_shoulder)



                # if mp_pose.PoseLandmark.LEFT_HIP.value in landmarks and mp_pose.PoseLandmark.RIGHT_HIP.value in landmarks:

                # center_x = (left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) / 4
                # center_y = (left_shoulder[1] + right_shoulder[1] + left_hip[1] + right_hip[1]) / 4
                # print(center_x, center_y, left_hip, right_hip)
                # cv2.circle(image, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)

                landmark_points = [(lm.x * width, lm.y * height) for lm in landmarks]

                # 计算人体中心点的坐标，即所有关键点的平均位置
                center_x = sum([pt[0] for pt in landmark_points]) / len(landmark_points)
                center_y = sum([pt[1] for pt in landmark_points]) / len(landmark_points)
                print("人体中心点坐标：", center_x, center_y)
                cv2.circle(image, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)



                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            #out.write(image)#检测视频保存，默认是保存在代码路径下
            cv2.imshow('Pose', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    #out.release()
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()

