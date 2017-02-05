import cv2
from sklearn.datasets import fetch_mldata
import numpy as np
from random import randint
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def get_trained_knn():
    print("Training k-NN classifier for MNIST dataset")
    mnist = fetch_mldata("MNIST original")
    KNN = cv2.ml.KNearest_create()
    traindata, trainlabels = [], []
    
    # populate labels
    for k in mnist.target:
        trainlabels.append(k)

    # populate images
    for d in mnist.data:
        traindata.append(np.array(d, dtype=np.float32))

    # train the model
    KNN.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels, dtype=np.int32))
    # KNN.save("hwdigits.xml")
    return KNN
