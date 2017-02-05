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
NUM_FEATURES = 20
FEAT_LEN = NUM_FEATURES * 128

detector, extractor = cv2.xfeatures2d.SIFT_create(NUM_FEATURES), cv2.xfeatures2d.SIFT_create(NUM_FEATURES)

def get_image(clss, num):
    return cv2.imread(path.join("..", "..", "CarData", "TrainImages", "%s-%d.pgm" % (clss, num)))

def extract(img):
    return np.ravel(cv2.goodFeaturesToTrack(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 8, 0.08, 3))

pos_samples = [extract(get_image(POS_CLASS, x)) for x in range(1, NUM_SAMPLES)]
neg_samples = [extract(get_image(NEG_CLASS, x)) for x in range(1, NUM_SAMPLES)]

all_samples = np.concatenate((pos_samples, neg_samples), axis = 0)
all_labels = np.concatenate((np.ones(len(pos_samples), dtype=np.int32), np.zeros(len(neg_samples), dtype=np.int32)), axis=0)

X_train, X_test, y_train, y_test = train_test_split(all_samples, all_labels, test_size=0.2, random_state=42)

NB = cv2.ml.NormalBayesClassifier_create()
NB.train(np.array(X_train, dtype=np.float32), cv2.ml.ROW_SAMPLE, np.array(y_train, dtype=np.int32))

y_pred = []

for i, d in enumerate(X_test):
    print i
    predicted = NB.predictProb(np.array([d], dtype=np.float32))[1][0][0]
    y_pred.append(predicted)

target_names = [POS_CLASS, NEG_CLASS]
print(classification_report(y_test, y_pred, target_names=target_names))


cnf_matrix = confusion_matrix(y_test, y_pred)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()

plot_confusion_matrix(cnf_matrix, classes=target_names,
                              title='Confusion matrix, without normalization')

plt.show()

print len(pos_samples[0])
