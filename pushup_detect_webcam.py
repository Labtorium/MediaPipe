import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose

def get_keypoint(results, height, width):
    left_shoulder_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width)
    left_shoulder_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height)
    left_shoulder_xy = [left_shoulder_x, left_shoulder_y]

    left_elbow_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * width)
    left_elbow_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * height)
    left_elbow_xy = [left_elbow_x, left_elbow_y]

    left_wrist_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * width)
    left_wrist_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * height)
    left_wrist_xy = [left_wrist_x, left_wrist_y]

    left_hip_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * width)
    left_hip_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * height)
    left_hip_xy = [left_hip_x, left_hip_y]

    left_knee_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * width)
    left_knee_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * height)
    left_knee_xy = [left_knee_x, left_knee_y]

    left_ankle_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x * width)
    left_ankle_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * height)
    left_ankle_xy = [left_ankle_x, left_ankle_y]
    return left_shoulder_xy, left_elbow_xy, left_wrist_xy, left_hip_xy, left_knee_xy, left_ankle_xy


if __name__ == '__main__':
    cap = cv2.VideoCapture('movies/push-up.mp4')

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        static_image_mode=False) as pose_detection:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height = rgb_frame.shape[0]
            width = rgb_frame.shape[1]

            results = pose_detection.process(rgb_frame)
            if not results.pose_landmarks:
                print('no result')
            else:
                left_shoulder_xy, left_elbow_xy, left_wrist_xy, left_hip_xy, left_knee_xy, left_ankle_xy = get_keypoint(results, height, width)

            cv2.imshow('push-up', frame)
            if cv2.waitKey(33) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()