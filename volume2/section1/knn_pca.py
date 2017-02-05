import cv2
from sklearn.datasets import fetch_mldata
import numpy as np
from random import randint
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# load mnist
mnist = fetch_mldata('MNIST original')

traindata, trainlabels = [],[]

for k in mnist.target:
    trainlabels.append(k)

for d in mnist.data:
    traindata.append(np.array(d, dtype=np.float32))

KNN = cv2.ml.KNearest_create()

mean, eigenvectors = cv2.PCACompute(np.array(traindata), mean=np.array([]), retainedVariance=0.5)

print eigenvectors

