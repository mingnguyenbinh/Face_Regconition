import glob
import cv2
from mtcnn.mtcnn import MTCNN

detector = MTCNN()
filenames = glob.glob("/home/minh/PycharmProjects/Untitled Folder/faces_raw/Pepe/*")
filenames.sort()
images = [cv2.imread(img) for img in filenames]
# 8GB Ram moi chay dc neu ko bi loi 137 sigkill 9
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
        face_in_img = cv2.resize(face_in_img, (160, 160))
        cv2.imwrite("/home/minh/PycharmProjects/Untitled Folder/faces_extracted/Pepe/" + str(i) + ".jpg", face_in_img)
        i += 1
