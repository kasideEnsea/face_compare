import cv2
import mediapipe as mp
import face_recognition
import pickle
import cv2
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from sklearn.neighbors import KDTree
import time
import os


def min_max_bounds(value, limit):
    if value > limit:
        value = limit
    elif value < 0:
        value = 0
    return int(value)


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

data = pickle.loads(open('face_enc', "rb").read())
tree = KDTree(data['encodings'], leaf_size=2, metric='euclidean')
f = open('max_distance.txt', 'r')
max_distance = float(f.read())
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            for detection in results.detections:
                coordinates = detection.location_data.relative_bounding_box
                xmin = min_max_bounds(coordinates.xmin * image.shape[1], image.shape[1])
                ymin = min_max_bounds(coordinates.ymin * image.shape[0], image.shape[0])
                width = coordinates.width * image.shape[1]
                height = coordinates.height * image.shape[0]
                encodings = face_recognition.face_encodings(image, [(ymin, min_max_bounds(xmin + width, image.shape[1]),

                                                                     min_max_bounds(ymin + height, image.shape[0]),
                                                                     xmin)])

                encoding = encodings[0]
                enc = [encoding]
                dist, ind = tree.query(enc, k=1)
                name = "Unknown"
                if dist < max_distance:
                    name = data['names'][ind[0][0]]
                cv2.rectangle(image, (xmin, ymin), (min_max_bounds(xmin + width, image.shape[1]),
                                                    min_max_bounds(ymin + height, image.shape[0])), (0, 255, 0), 2)
                cv2.putText(image, name, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)
                # cv2.imshow("Frame", image)
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
