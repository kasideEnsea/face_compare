import face_recognition
import imutils
import pickle

import numpy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import time
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import sklearn.neighbors

from sklearn.neighbors import DistanceMetric

data = pickle.loads(open('face_enc', "rb").read())

standarted_data = StandardScaler().fit_transform(data["encodings"])
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(standarted_data)
fig = plt.figure(figsize=(8, 8))
plt.xlabel = 'Principal Component 1'
plt.ylabel = 'Principal Component 2'
plt.title = '2 component PCA'
plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c='red')
plt.show()

min_distances = []
for encoding in data['encodings']:
    distance = np.linalg.norm(data['encodings'] - encoding, axis=1)
    distance = sorted(distance)
    min_distances.append(distance[1])

plt.hlines(0, xmin=min(min_distances), xmax=max(min_distances))
plt.scatter(min_distances, [0] * len(min_distances), c='#0a0b0c3a', s=50)
plt.show()

plt.hist(min_distances, 6)
plt.show()

min_distances = sorted(min_distances)
noise_count = int(np.round(len(min_distances) * 0.025))
del min_distances[0:noise_count]
if len(min_distances) > 0:
    f = open('max_distance.txt', 'w')
    f.write(str(min_distances[0]))
else:
    f = open('max_distance.txt', 'w')
    f.write('0.55')
f.close()
