from modules import *
from MP_holistic_styled_landmarks import mp_holistic, draw_styled_landmarks
from folder_setup import *
from mediapipe_detection import mediapipe_detection
from keypoints_extraction import extract_keypoints
from PIL import ImageFont, ImageDraw, Image

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(200):
            #path = 'C:/Users/user/Desktop/졸업작품/3차 영상/' + action + '/' + str(sequence) +'.avi'
            #print(path)
            #out = cv2.VideoWriter(path, fourcc, 30, (1280, 720))
            print(sequence)
            for frame_num in range(sequence_length):
                ret, frame = cap.read()

                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                b, g, r, a = 1, 255, 0, 0
                fontpath = "fonts/gulim.ttc"
                font = ImageFont.truetype(fontpath, 20)
                img_pil = Image.fromarray(image)
                draw = ImageDraw.Draw(img_pil)
                draw.text((15, 12), action, font=font, fill=(b, g, r, a))
                image = np.array(img_pil)

                # Wait
                if frame_num == 0:
                    cv2.putText(image, 'START NEW ACTION', (120,200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                    cv2.imshow('Action Detection', image)
                    cv2.waitKey(500)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                    cv2.imshow('Action Detection', image)

                keypoints = extract_keypoints(results)
                keypoint_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(keypoint_path, keypoints)
                #out.write(frame)
                if cv2.waitKey(10) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


