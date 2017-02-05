import sklearn.svm as svm
import cv2
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import fetch_olivetti_faces as fetch

dataset = fetch()
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print np.unique(y_train)

svm = svm.SVC(kernel='linear', C=0.099)
parameters = {}
clf = GridSearchCV(svm, parameters, verbose=1)
clf.fit(X_train, y_train)
print clf.best_params_

y_pred = clf.predict(X_test)

print classification_report(y_test, y_pred)
