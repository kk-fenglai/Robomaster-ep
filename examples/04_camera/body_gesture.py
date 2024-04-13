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
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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

                # 获取3D坐标
                left_shoulder_3d = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z])
                left_elbow_3d = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z])
                left_wrist_3d = np.array([landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z])
                left_hip_3d = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z])
                right_shoulder_3d = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z])
                right_elbow_3d = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z])
                right_wrist_3d = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z])
                right_hip_3d = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z])

                # 计算3D角度
                angle_left_arm_body_x = calculate_3d_angle(
                    left_shoulder_3d, left_elbow_3d, left_hip_3d)
                angle_right_arm_body_x = calculate_3d_angle(
                    right_shoulder_3d, right_elbow_3d, right_hip_3d)
                angle_left_arm_body_y = calculate_3d_angle(
                    left_wrist_3d, left_elbow_3d, left_shoulder_3d)
                angle_right_arm_body_y = calculate_3d_angle(
                    right_wrist_3d, right_elbow_3d, right_shoulder_3d)
                angle_left_arm_forearm = calculate_3d_angle(
                    left_shoulder_3d, left_elbow_3d, left_wrist_3d)
                angle_right_arm_forearm = calculate_3d_angle(
                    right_shoulder_3d, right_elbow_3d, right_wrist_3d)
                angle_head_forearm = calculate_angle(left_eye, right_eye, nose)

                # 高斯滤波
                angle_history["left_arm_body_x"] = np.append(
                    angle_history["left_arm_body_x"][1:], angle_left_arm_body_x)
                angle_history["right_arm_body_x"] = np.append(
                    angle_history["right_arm_body_x"][1:], angle_right_arm_body_x)
                angle_history["left_arm_body_y"] = np.append(
                    angle_history["left_arm_body_y"][1:], angle_left_arm_body_y)
                angle_history["right_arm_body_y"] = np.append(
                    angle_history["right_arm_body_y"][1:], angle_right_arm_body_y)
                angle_history["left_arm_forearm"] = np.append(
                    angle_history["left_arm_forearm"][1:], angle_left_arm_forearm)
                angle_history["right_arm_forearm"] = np.append(
                    angle_history["right_arm_forearm"][1:], angle_right_arm_forearm)
                angle_history["head_forearm"] = np.append(
                    angle_history["head_forearm"][1:], angle_head_forearm)

                sigma = 10
                angle_left_arm_body_x_filtered = gaussian_filter(
                    angle_history["left_arm_body_x"], sigma=sigma, mode='reflect')[-1]
                angle_right_arm_body_x_filtered = gaussian_filter(
                    angle_history["right_arm_body_x"], sigma=sigma, mode='reflect')[-1]
                angle_left_arm_body_y_filtered = gaussian_filter(
                    angle_history["left_arm_body_y"], sigma=sigma, mode='reflect')[-1]
                angle_right_arm_body_y_filtered = gaussian_filter(
                    angle_history["right_arm_body_y"], sigma=sigma, mode='reflect')[-1]
                angle_left_arm_forearm_filtered = gaussian_filter(
                    angle_history["left_arm_forearm"], sigma=sigma, mode='reflect')[-1]
                angle_right_arm_forearm_filtered = gaussian_filter(
                    angle_history["right_arm_forearm"], sigma=sigma, mode='reflect')[-1]
                angle_head_forearm_filtered = gaussian_filter(
                    angle_history["head_forearm"], sigma=sigma, mode='reflect')[-1]

                cv2.putText(image, f"X-axis: {int(angle_left_arm_body_x_filtered)}",
                            (int(left_shoulder[0]), int(left_shoulder[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"X-axis: {int(angle_right_arm_body_x_filtered)}",
                            (int(right_shoulder[0]), int(right_shoulder[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"Y-axis: {int(angle_left_arm_body_y_filtered)}",
                            (int(left_shoulder[0]), int(left_shoulder[1] + 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"Y-axis: {int(angle_right_arm_body_y_filtered)}",
                            (int(right_shoulder[0]), int(right_shoulder[1] + 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, f"{int(angle_left_arm_forearm_filtered)}", (int(left_elbow[0]), int(
                    left_elbow[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"{int(angle_right_arm_forearm_filtered)}", (int(right_elbow[0]), int(
                    right_elbow[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"Head: {int(angle_head_forearm_filtered)}", (int(nose[0]), int(
                    nose[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                # 手臂相对于身体的X轴（上举/下垂）
                # angle_left_arm_body_x_filtered = 左臂身体角度 x (高斯滤波版)
                # angle_right_arm_body_x_filtered = 右臂身体角度 x (高斯滤波版)
                #
                # 手臂相对于身体的Y轴（展开/收紧）
                # angle_left_arm_body_y_filtered = 左臂身体角度 y (高斯滤波版)
                # angle_right_arm_body_y_filtered = 右臂身体角度 y (高斯滤波版)
                #
                # 前臂（小手臂和大手臂）角度 （展开/收紧）
                # angle_left_arm_forearm_filtered = 左臂前臂角度 (高斯滤波版)
                # angle_right_arm_forearm_filtered = 右臂前臂角度 (高斯滤波版)
                #
                # 头部角度
                # angle_head_forearm_filtered = 头部角度 (高斯滤波版)

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

