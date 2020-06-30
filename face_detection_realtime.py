import cv2
import time
from mtcnn.mtcnn import MTCNN
import math
import numpy as np
from keras.models import load_model
import joblib
import firebase_admin
from firebase_admin import credentials, firestore, storage, db
from sklearn import svm, metrics, preprocessing
import string
import random

out_encoder = preprocessing.LabelEncoder()
label = np.load('label.npy')
out_encoder.fit(label)

joblib_model = joblib.load('svm_model.pkl')
model = load_model('facenet_keras.h5', compile=False)
cred = credentials.Certificate("/home/minh/Downloads/project-3d6b7-firebase-adminsdk-vrh3t-c5123e378f.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'project-3d6b7.appspot.com',
    'databaseURL': 'https://project-3d6b7.firebaseio.com/'
})
def randomString(stringLength=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def get_face_embedded(model, image):
    mean, std = image.mean(), image.std()
    image = (image - mean) / std
    sample = np.expand_dims(image, axis=0)
    y = model.predict(sample)
    return y[0]


font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)
detector = MTCNN()
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))
frame_id = 0
n=0
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_new=frame.copy()
    # run faster
    if math.fmod(frame_id, 5) == 0:
        # start_time = time.time()
        result = detector.detect_faces(frame)
        # elapsed_time = time.time() - start_time
        # print('inference time cost: {}'.format(elapsed_time))
    if result:
        for person in result:
            bounding_box = person['box']
            x = bounding_box[0]
            y = bounding_box[1]
            w = bounding_box[0] + bounding_box[2]
            h = bounding_box[1] + bounding_box[3]
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 3)
            face_in_img = frame[y:h, x:w]
            if face_in_img is not None:
                try:
                    face_in_img = cv2.resize(face_in_img, (160, 160),interpolation=cv2.INTER_AREA)
                    face_in_img = face_in_img.astype('float32')
                    X = get_face_embedded(model, face_in_img)
                    X = np.reshape(X, (1, -1))
                    Y = joblib_model.predict(X)
                    prob=joblib_model.decision_function(X)
                    prob_max=np.max(prob)
                    prob_max=np.exp(prob_max)/np.sum(np.exp(prob))
                    print(prob_max)
                    predict_names = out_encoder.inverse_transform(Y)
                    cv2.putText(frame, predict_names[0], (10, 450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
                    if predict_names[0]=='Minh' and prob_max>0.7:
                        ref = db.reference('TKPgrEeAUKUShLKgKGWG7dIktCT2/Device/')
                        ref.update({'Gate': 1})
                    else:
                        img_item = "strange.png"
                        # cv2.imwrite(img_item, frame_new)
                        # bucket = storage.bucket()
                        # path = randomString()
                        # blob = bucket.blob('TKPgrEeAUKUShLKgKGWG7dIktCT2/' + path + '.png')
                        # outfile = '/home/minh/PycharmProjects/Untitled Folder/strange.png'
                        # blob.upload_from_filename(outfile)
                except:
                    print()
                # print(Y)
    cv2.imshow('frame', frame)
    frame_id += 1
    # out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# out.release()
cap.release()
cv2.destroyAllWindows()
