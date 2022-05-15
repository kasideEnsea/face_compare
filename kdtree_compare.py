import face_recognition
import pickle
import cv2
from sklearn.neighbors import KDTree
import time
import os

'''cascPathface = "haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPathface)
data = pickle.loads(open('face_enc', "rb").read())
tree = KDTree(data['encodings'], leaf_size=2, metric='euclidean')
f = open('max_distance.txt', 'r')
max_distance = float(f.read())
image = cv2.imread("ref/alexander.jpg")
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray,
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(60, 60),
                                     flags=cv2.CASCADE_SCALE_IMAGE)

encodings = face_recognition.face_encodings(rgb)
names = []

for encoding in encodings:
    enc = [encoding]
    dist, ind = tree.query(enc, k=1)
    if dist < max_distance:
        print(data['names'][ind[0][0]])
    else:
        print("Unknown")'''

cascPathface = "haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPathface)
data = pickle.loads(open('face_enc', "rb").read())
tree = KDTree(data['encodings'], leaf_size=2, metric='euclidean')
f = open('max_distance.txt', 'r')
max_distance = float(f.read())
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    names = []
    # grab the frame from the threaded video stream
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    for encoding in encodings:
        enc = [encoding]
        dist, ind = tree.query(enc, k=1)
        name = "Unknown"
        if dist < max_distance:
            name = data['names'][ind[0][0]]
        print(name)
        names.append(name)
    for ((x, y, w, h), name) in zip(faces, names):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()