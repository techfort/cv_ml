from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils import np_utils
from imutils import paths
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import argparse
import cv2
import os

target_names = ["daffodil", "snowdrop", "lily of the valley", "bluebell"]
NUM_SAMPLES = 80
NUM_CLASSES = len(target_names)
SIZE=32

def image_to_vector(image, size=(32,32)):
    return cv2.resize(image, size)

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
X_train = X_train.reshape(X_train.shape[0], 3, SIZE, SIZE)
X_test = X_test.reshape(X_test.shape[0], 3, SIZE, SIZE)

def create_model(learning_rate=0.1, momentum=0.9):
    model = Sequential()
    model.add(Convolution2D(20, 9, 9, border_mode='same', input_shape=(3, SIZE, SIZE)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Convolution2D(50, 5, 5, activation = "relu"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(768, input_dim=3072, init='uniform', activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(384, init = 'uniform',  activation = 'relu', W_constraint=maxnorm(3)))
    model.add(Dense(4))
    model.add(Activation("softmax"))
    sgd = SGD(lr=learning_rate, momentum=momentum, nesterov=True, decay=1e-6)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])
    return model

model = KerasClassifier(build_fn=create_model, nb_epoch=20, batch_size=10, verbose=1)
lr = [0.01, 0.1, 0.5]
mmnt = [0.7, 0.8, 0.9]
param_grid = dict(learning_rate=lr, momentum=mmnt)
grid = GridSearchCV(estimator=model, param_grid=param_grid)

grid.fit(X_train, y_train) 
print grid.best_params_
# result: learning_rate = 0.01, momentum = 0.9
y_pred = grid.predict(y_test)
print classification_report(y_test, y_pred)
# (loss, accuracy) = grid.evaluate(X_test, y_test, verbose = 1)
# print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
