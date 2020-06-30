import cv2
from mtcnn.mtcnn import MTCNN
import math
detector = MTCNN()
cap = cv2.VideoCapture(0)
i=10
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    result = detector.detect_faces(frame)
    if result:
        bounding_box = result[0]['box']
        x = bounding_box[0]
        y = bounding_box[1]
        w = bounding_box[0] + bounding_box[2]
        h = bounding_box[1] + bounding_box[3]
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 3)
        face_in_img = frame[y:h, x:w]
        if cv2.waitKey(1) & 0xFF == ord('s'):
            if face_in_img is not None:
                face_in_img = cv2.resize(face_in_img, (160, 160))
                cv2.imwrite("faces_extracted\\Minh\\" + str(i) + ".jpg", face_in_img)
                i+=1
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()