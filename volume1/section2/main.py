import lib.training as td 
import cv2
import lib.sift as sift
import numpy as np


data = td.getTrainingData()
raw = data["raw"]
train = data["traindata"]
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
svm = cv2.ml.SVM_create();
svm.setGamma(0.2)
svm.setC(30)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setTermCriteria(criteria)
svm.train(train)

predictions = {}

for image in raw:
    ft = np.array(sift.extract_features(image["img"]), dtype=np.float32)
    if len(ft) > td.NUM_FEATURES:
        ft = ft[0:td.NUM_FEATURES]
    if len(ft) == td.NUM_FEATURES:
        predict = svm.predict(np.array([ft]))[1][0][0]
        if predict in predictions:
            predictions[predict] += 1
        else:
            predictions[predict] = 1
    else:
        print "expected %d but got %d" % (td.NUM_FEATURES, len(ft))

print predictions

