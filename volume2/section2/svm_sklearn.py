import sklearn.svm as svm
import cv2
import numpy as np
import argparse
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def rng(x):
    return range(1, x + 1)

def read(cls, id):
    return np.ravel(cv2.imread(img_path % (BASE_PATH, cls, id), 0))

NUM_CLASSES = 20 
NUM_SAMPLES = 10
BASE_PATH = "../../orl_faces/"
img_path = "%s/s%d/%d.pgm"
target_names = ["face_%d" % x for x in rng(NUM_CLASSES)]
X, y = [], []

print cv2.imread(img_path % (BASE_PATH, 1, 1), 0).shape

for x in rng(NUM_CLASSES):
    for i in rng(NUM_SAMPLES):
        X.append(read(x, i))
        y.append(x)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=42)

svm = svm.LinearSVC()
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

print(classification_report(y_test, y_pred, target_names=target_names))

for x in range(1,20):
    img = X_test[x]
    cls = y_test[x]
    pred = y_pred[x]
    cv2.imshow("Face: %d, Prediction: %d" % (cls, pred), np.reshape(img, (112, 92)))
    cv2.waitKey()

