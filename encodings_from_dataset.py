from imutils import paths
import pickle
import face_recognition
import cv2
import os
from os import path

imagePaths = list(paths.list_images('Images'))
knownEncodings = []
knownNames = []
for (i, imagePath) in enumerate(imagePaths):
    name = path.splitext(path.basename(imagePath))[0]
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)
data = {"encodings": knownEncodings, "names": knownNames}
f = open("face_enc", "wb")
f.write(pickle.dumps(data))
f.close()