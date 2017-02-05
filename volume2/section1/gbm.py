import cv2
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os.path as path
from plot_cm import plot_confusion_matrix

NUM_SAMPLES = 499
POS_CLASS = "pos"
NEG_CLASS = "neg"
NUM_FEATURES = 20

def get_image(clss, num):
    return cv2.imread(path.join("..", "..", "CarData", "TrainImages", "%s-%d.pgm" % (clss, num)))

def extract(img):
    return np.ravel(cv2.calcHist([img],[2],None,[127],[0,256]))

pos_samples = [extract(get_image(POS_CLASS, x)) for x in range(1, NUM_SAMPLES)]
neg_samples = [extract(get_image(NEG_CLASS, x)) for x in range(1, NUM_SAMPLES)]

all_samples = np.concatenate((pos_samples, neg_samples), axis = 0)
all_labels = np.concatenate((np.ones(len(pos_samples), dtype=np.int32), np.zeros(len(neg_samples), dtype=np.int32)), axis=0)
all_samples = all_samples / 4000.
X_train, X_test, y_train, y_test = train_test_split(all_samples, all_labels, test_size=0.2, random_state=42)

clf = GradientBoostingClassifier(n_estimators=1000, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

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
