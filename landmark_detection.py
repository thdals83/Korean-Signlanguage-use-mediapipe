
from modules import *
from mediapipe_detection import mediapipe_detection
from MP_holistic_styled_landmarks import draw_styled_landmarks, mp_holistic

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret,frame = cap.read()
        
        image,results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image,results)
        cv2.imshow('check',image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
