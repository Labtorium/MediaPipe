import mediapipe as mp
import cv2

mp_face_detect = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

kp_drawing_spec = mp_drawing.DrawingSpec(thickness=3, circle_radius=3, color=(0, 255, 0))
bbox_drawing_spec = mp_drawing.DrawingSpec(thickness=3, color=(0, 0, 255))

img_path = 'face.jpg'

with mp_face_detect.FaceDetection(min_detection_confidence=0.5) as face_detection:
    image = cv2.imread(img_path)
    image = cv2.resize(image, dsize=None, fx=0.5, fy=0.5)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height = rgb_image.shape[0]
    width = rgb_image.shape[1]

    results = face_detection.process(rgb_image)

    annotated_image = image.copy()
    for detection in results.detections:
        print('--- original coordinate')
        print('Nose tip:')
        print(mp_face_detect.get_key_point(detection, mp_face_detect.FaceKeyPoint.NOSE_TIP))
        nose_x = mp_face_detect.get_key_point(detection, mp_face_detect.FaceKeyPoint.NOSE_TIP).x
        nose_y = mp_face_detect.get_key_point(detection, mp_face_detect.FaceKeyPoint.NOSE_TIP).y
        print('--- convert coordinate')
        re_nose_x = int(nose_x * width)
        re_nose_y = int(nose_y * height)
        print('x:', re_nose_x, 'y:', re_nose_y)

        mp_drawing.draw_detection(annotated_image, detection, keypoint_drawing_spec=kp_drawing_spec, bbox_drawing_spec=bbox_drawing_spec)
        cv2.imwrite('result.jpg', annotated_image)
        
