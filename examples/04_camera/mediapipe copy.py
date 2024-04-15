import sys
import cv2
import time
import mediapipe as mp
 
mp_pose = mp.solutions.pose
drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    enable_segmentation=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
 
 
def process_frame(img):
    results = pose.process(img)
    drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return img
 
 
if __name__ == '__main__':
    t0 = time.time()
    cap = cv2.VideoCapture(0)
    cap.open(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            raise ValueError("Error")
        frame = process_frame(frame)
        cv2.imshow("keypoint", frame)
        if ((time.time() - t0) // 1) == 10:
            sys.exit(0)
        cv2.waitKey(1)
 
    cap.release()
    cv2.destroyAllWindows()