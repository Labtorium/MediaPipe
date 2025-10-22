# pylint: disable=no-member
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

mesh_drawing_spec = mp_drawing.DrawingSpec(thickness=3, color=(0, 255, 0))
mark_drawing_spec = mp_drawing.DrawingSpec(thickness=3, circle_radius=3, color=(0, 0, 255))

img_path = 'images/pose.jpg'

with mp_pose.Pose(
    min_detection_confidence=0.3,
    static_image_mode=True) as pose_detection:

    image = cv2.imread(img_path)
    image = cv2.resize(image, dsize=None, fx=0.5, fy=0.5)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height = rgb_image.shape[0]
    width = rgb_image.shape[1]

    results = pose_detection.process(rgb_image)

    annotated_image = image.copy()
    print('left foot index x:', results.pose_landmarks.landmark[31].x)
    print('left foot index y:', results.pose_landmarks.landmark[31].y)
    print('right foot index x:', results.pose_landmarks.landmark[32].x)
    print('right foot index y:', results.pose_landmarks.landmark[32].y)
    print('right ankle x:', results.pose_landmarks.landmark[28].x)
    print('right ankle y:', results.pose_landmarks.landmark[28].y)
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=results.pose_landmarks,
        connections=mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mark_drawing_spec,
        connection_drawing_spec=mesh_drawing_spec
    )

    cv2.imwrite('result.jpg', annotated_image)
