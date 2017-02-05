import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os.path as path
from plot_cm import plot_confusion_matrix

NUM_SAMPLES = 499
POS_CLASS = "pos"
NEG_CLASS = "neg"
NUM_FEATURES = 25
FEAT_LEN = NUM_FEATURES * 128

detector, extractor = cv2.xfeatures2d.SIFT_create(NUM_FEATURES), cv2.xfeatures2d.SIFT_create(NUM_FEATURES)

def get_image(clss, num):
    return cv2.imread(path.join("..", "..", "CarData", "TrainImages", "%s-%d.pgm" % (clss, num)))

def get_features(img, det, extr):
    """this returns keypoints and descriptor, we only need descriptors"""
    return np.ravel(np.array(extr.compute(img, det.detect(img))[1], dtype=np.float32))

def get_extractor(det, extr):
    def ext(img):
        return get_features(img, det, extr)
    return ext

def compress(features):
    mean, eigenvectors = cv2.PCACompute(features, mean=np.array([]), maxComponents=150) 
    return np.ravel(cv2.PCAProject(features, mean, eigenvectors))

"""get the curried extract function"""
extract = get_extractor(detector, extractor)

pos_samples = [extract(get_image(POS_CLASS, x)) for x in range(1, NUM_SAMPLES)]
neg_samples = [extract(get_image(NEG_CLASS, x)) for x in range(1, NUM_SAMPLES)]

pos_samples = [compress(x) for x in pos_samples if len(x) == FEAT_LEN]
neg_samples = [compress(x) for x in neg_samples if len(x) == FEAT_LEN] 

all_samples = np.concatenate((pos_samples, neg_samples), axis = 0)
all_labels = np.concatenate((np.ones(len(pos_samples), dtype=np.int32), np.zeros(len(neg_samples), dtype=np.int32)), axis=0)

X_train, X_test, y_train, y_test = train_test_split(all_samples, all_labels, test_size=0.2, random_state=42)

LR = cv2.ml.LogisticRegression_create()
"""learning rate alpha, 0.1 is quite conservative """
LR.setLearningRate(0.00001)
""" testing L1 """
LR.setRegularization(cv2.ml.LOGISTIC_REGRESSION_REG_L1)
""" testing mini-batch """
LR.setTrainMethod(cv2.ml.LogisticRegression_BATCH)
LR.setMiniBatchSize(110)
""" setting 1000 iterations """
LR.setIterations(3000)
"""train"""
LR.train(np.array(X_train, dtype=np.float32), cv2.ml.ROW_SAMPLE, np.array(y_train, dtype=np.float32))

y_pred = []

for i, d in enumerate(X_test):
    print i
    predicted = LR.predict(np.array([d], dtype=np.float32))[1][0][0]
    y_pred.append(predicted)

target_names = [POS_CLASS, NEG_CLASS]
print(classification_report(y_test, y_pred, target_names=target_names))


cnf_matrix = confusion_matrix(y_test, y_pred)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=target_names,
                              title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
                              title='Normalized confusion matrix')

plt.show()
