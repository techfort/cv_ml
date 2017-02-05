import cv2
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os.path as path
from plot_cm import plot_confusion_matrix

NUM_SAMPLES = 499
POS_CLASS = "pos"
NEG_CLASS = "neg"
NUM_FEATURES = 40
RANDOM_STATE = 42
MAX_DEPTH = 21 

def get_image(clss, num):
    return cv2.imread(path.join("..", "..", "CarData", "TrainImages", "%s-%d.pgm" % (clss, num)))

def extract(img):
    return np.ravel(cv2.calcHist([img],[2],None,[126],[0,256]))

pos_samples = [extract(get_image(POS_CLASS, x)) for x in range(1, NUM_SAMPLES)]
neg_samples = [extract(get_image(NEG_CLASS, x)) for x in range(1, NUM_SAMPLES)]

all_samples = np.concatenate((pos_samples, neg_samples), axis = 0)
all_labels = np.concatenate((np.ones(len(pos_samples), dtype=np.int32), np.zeros(len(neg_samples), dtype=np.int32)), axis=0)
all_samples = all_samples / 4000.
X_train, X_test, y_train, y_test = train_test_split(all_samples, all_labels, test_size=0.2, random_state=42)

target_names = [POS_CLASS, NEG_CLASS]

print("DecisionTreeClassifier >>>>>")
dt = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=MAX_DEPTH)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print(classification_report(y_test, y_pred, target_names=target_names))

rfc = RandomForestClassifier(n_estimators=60,max_features=NUM_FEATURES, random_state=RANDOM_STATE)
rfc.fit(X_train, y_train)
print("RandomForestClassifier >>>>>")
rfc_pred = rfc.predict(X_test)
print classification_report(y_test, rfc_pred, target_names=target_names)

gbm = GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=59, learning_rate=0.1, max_depth=MAX_DEPTH, subsample=0.7)
gbm.fit(X_train, y_train)
print("GradientBoostingClassifier >>>>>>")
gbm_pred = gbm.predict(X_test)
print classification_report(y_test, gbm_pred, target_names=target_names)

ada = AdaBoostClassifier(base_estimator=dt, random_state=RANDOM_STATE)
ada.fit(X_train, y_train)
print("AdaBoostClassifier >>>>>")
ada_pred = ada.predict(X_test)
print classification_report(ada_pred, y_test, target_names=target_names)

bag = BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=MAX_DEPTH, max_features=NUM_FEATURES), n_estimators=70, max_features=NUM_FEATURES)
bag.fit(X_train, y_train)
print("BaggingClassifier >>>>>>")
bag_pred = bag.predict(X_test)
print classification_report(bag_pred, y_test, target_names=target_names)

