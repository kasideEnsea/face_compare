import face_recognition
import imutils
import pickle
import time
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import numpy as np
from datetime import datetime
import random

classification_time_dlib = []
accuracy_check_dlib = []
classification_time = []
accuracy_check = []
cascPathface = "haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPathface)
data = pickle.loads(open('face_enc', "rb").read())
tree = KDTree(data['encodings'], leaf_size=2, metric='euclidean')
f = open('max_distance.txt', 'r')
max_distance = float(f.read())
min_metrika = min(map(min, data["encodings"]))
max_metrika = max(map(max, data["encodings"]))
name = "andrey"
experiment_sizes = []
for i in range (50, 3000):
    new_encoding = []
    for j in range (128):
        new_encoding.append(round(random.uniform(min_metrika, max_metrika), 10))
    data['encodings'].append(new_encoding)

image = cv2.imread("ref/" + name + ".jpg")
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)

encodings = face_recognition.face_encodings(rgb)

for i in range(51, len(data['encodings']), 50):
    experiment_sizes.append(i)
    current_encoding = data['encodings'][:i]
    tree = KDTree(current_encoding, leaf_size=2, metric='euclidean')
    start = datetime.now()
    dist, ind = tree.query(encodings, k=1)
    classified_name = "Unknown"
    if dist < max_distance:
        classified_name = data['names'][ind[0][0]]
    finish = datetime.now()
    classification_time.append((finish - start).total_seconds())
    accuracy_check.append(classified_name == name)

    for encoding in encodings:
        matches = face_recognition.compare_faces(current_encoding,
                                                 encoding)
        classified_name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                classified_name = data["names"][i]
                counts[classified_name] = counts.get(classified_name, 0) + 1
                classified_name = max(counts, key=counts.get)
        finish = datetime.now()
        classification_time_dlib.append((finish - start).total_seconds())
        accuracy_check_dlib.append(classified_name == name)

print(accuracy_check)
print(accuracy_check_dlib)

fig = plt.figure(figsize=(8, 8))
plt.xlabel = 'Размер датасета'
plt.ylabel = 'Время'
plt.title = 'Сравнение времени работы dlib и kd-tree на больших массивах'
plt.plot(experiment_sizes, classification_time, c='blue')
plt.plot(experiment_sizes, classification_time_dlib, c='red')
plt.show()

data = pickle.loads(open('face_enc', "rb").read())
tree = KDTree(data['encodings'], leaf_size=2, metric='euclidean')
names = ['alexander', 'sergey', 'andrey', 'anna', 'dmitry', 'ekaterina', 'evelina', 'olga', 'polina', 'ruslan']
classification_time_dlib = []
accuracy_check_dlib = []
classification_time = []
accuracy_check = []
for name in names:
    image = cv2.imread("ref/" + name + ".jpg")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)

    encodings = face_recognition.face_encodings(rgb)
    start = datetime.now()
    dist, ind = tree.query(encodings, k=1)
    classified_name = "Unknown"
    if dist < max_distance:
        classified_name = data['names'][ind[0][0]]
    finish = datetime.now()
    classification_time.append((finish - start).total_seconds())
    accuracy_check.append(classified_name == name)

    encodings = face_recognition.face_encodings(rgb)
    start = datetime.now()
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"],
                                                 encoding)
        classified_name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                classified_name = data["names"][i]
                counts[classified_name] = counts.get(classified_name, 0) + 1
                classified_name = max(counts, key=counts.get)
        finish = datetime.now()
        classification_time_dlib.append((finish - start).total_seconds())
        accuracy_check_dlib.append(classified_name == name)

print(classification_time_dlib)
print(classification_time)
print(accuracy_check)
print(accuracy_check_dlib)

fig = plt.figure(figsize=(8, 8))
plt.xlabel = 'Experiment'
plt.ylabel = 'Time'
plt.title = 'Сравнение времени работы dlib и kd-tree'
plt.plot(names, classification_time, c='blue')
plt.plot(names, classification_time_dlib, c='red')
plt.show()
