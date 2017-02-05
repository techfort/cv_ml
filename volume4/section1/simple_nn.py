from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils import np_utils
from imutils import paths
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import argparse
import cv2
import os

target_names = ["daffodil", "snowdrop", "lily of the valley", "bluebell"]
NUM_SAMPLES = 80
NUM_CLASSES = len(target_names)

def image_to_vector(image, size=(32,32)):
    return cv2.resize(image, size).flatten()

def label_vec(x):
    res = np.zeros(4)
    res[x] = 1
    return res

data, labels = [], []
counter = 1
for i in range(0, NUM_CLASSES):
    for j in range(1, NUM_SAMPLES + 1):
        data.append(image_to_vector(cv2.imread("../../flowers/jpg/image_%04d.jpg" % counter)))
        labels.append(target_names[i])
        if counter % 20 == 0:
            print "[INFO] processed {}/{}".format(counter, NUM_CLASSES * NUM_SAMPLES)
        counter += 1

data = np.array(data) / 255.0
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels, 4)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(768, input_dim=3072))
model.add(Activation("relu"))
model.add(Dense(384))
model.add(Activation("relu"))
model.add(Dense(4))
model.add(Activation("softmax"))

sgd = SGD(lr=0.01, momentum=0.9, nesterov=True, decay=1e-6)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])
model.fit(X_train, y_train, batch_size=40, nb_epoch=1000) 

(loss, accuracy) = model.evaluate(X_test, y_test, verbose = 1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
