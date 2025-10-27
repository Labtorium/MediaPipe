import mediapipe as mp
import cv2
import numpy as np

THRESH_SLOP = 0
THRESH_DIST_SPINE = 30
THRESH_ARM = 40

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

def calc_distance(x1, y1, x2, y2, x3, y3):
    u = np.array([x2 - x1, y2 - y1])
    v = np.array([x3 - x1, y3 - y1])
    L = abs(np.cross(u, v)) / np.linalg.norm(u)
    return L

def calc_slope(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)
    return slope

def is_low_pose(dist_hip, dist_knee, dist_elbow, body_slope):
    if dist_hip < THRESH_DIST_SPINE and dist_knee < THRESH_DIST_SPINE and dist_elbow > THRESH_ARM and body_slope <= THRESH_SLOP:
        return True
    else:
        return False


if __name__ == '__main__':
    #cap = cv2.VideoCapture(0)  ## For webcam input
    cap = cv2.VideoCapture('movies/push-up.mp4')
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS))

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        static_image_mode=False) as pose_detection:

        flg_low = False
        count = 0

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
                print('No pose landmarks detected')
            else:
                left_shoulder_xy, left_elbow_xy, left_wrist_xy, left_hip_xy, left_knee_xy, left_ankle_xy = get_keypoint(results, height, width)
                dist_hip = calc_distance(left_shoulder_xy[0], left_shoulder_xy[1], left_ankle_xy[0], left_ankle_xy[1], left_hip_xy[0], left_hip_xy[1])
                dist_knee = calc_distance(left_shoulder_xy[0], left_shoulder_xy[1], left_ankle_xy[0], left_ankle_xy[1], left_knee_xy[0], left_knee_xy[1])
                dist_elbow = calc_distance(left_shoulder_xy[0], left_shoulder_xy[1], left_wrist_xy[0], left_wrist_xy[1], left_elbow_xy[0], left_elbow_xy[1])
                body_slope = calc_slope(left_shoulder_xy[0], left_shoulder_xy[1], left_ankle_xy[0], left_ankle_xy[1])

                prev_flg = flg_low
                flg_low = is_low_pose(dist_hip, dist_knee, dist_elbow, body_slope)
                if prev_flg == False and flg_low == True:
                    count += 1
                    print('Push-up detected')

                cv2.putText(frame, 'Push-Up Count: ' + str(count), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('push-up', frame)
            if cv2.waitKey(33) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()