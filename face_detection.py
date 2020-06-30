import glob
import cv2
from mtcnn.mtcnn import MTCNN

detector = MTCNN()
filenames = glob.glob("C:\\Users\\Minh\\Desktop\\Face_Regconition\\faces_raw\\Unknown\\*")
filenames.sort()
images = [cv2.imread(img) for img in filenames]
i = 0
for img in images:
    result = detector.detect_faces(img)
    for person in result:
        bounding_box = person['box']
        x = bounding_box[0]
        y = bounding_box[1]
        w = bounding_box[0] + bounding_box[2]
        h = bounding_box[1] + bounding_box[3]
        face_in_img = img[y:h, x:w]
        try:
            face_in_img = cv2.resize(face_in_img, (160, 160))
            cv2.imwrite("faces_extracted\\Unknown\\" + str(i) + ".jpg", face_in_img)
            i += 1
        except:
            print("error")
