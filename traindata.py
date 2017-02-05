# coding: utf-8

import numpy as np
samples = np.array([], dtype=np.float32)
np.append(samples, 1.)
samples
np.append(samples, 1., 2.)
np.append(samples, [1., 2.])
samples = [[1,1,1],[2,1,3],[2,2,2],[3,2,3],[4,3,7]]
labels = [1,2,2,3,3]
len(samples) == len(labels)
import cv2
help(cv2.ml)
lr = cv2.ml.LogisticRegression_create()
import numpy as np
lr.train(np.array(samples, dtype=np.float32), cv2.ml.ROW_SAMPLE, np.array(labels, dtype=np.float32))
lr.predict([[2,3,2]])
lr.predict(np.array([[2,3,2]], dtype=np.float32))
det, ext = cv2.xfeatures2d.SIFT_create(25), cv2.xfeatures2d.SIFT_create(25)
def get_features(img):
    return np.array(np.ravel(ext.compute(img, det.detect(img))[1]) ,dtype=np.float32)
smpls = [get_features(cv2.imread("CarData/TrainImages/pos-%d.pgm" % x)) for x in range(10,20)]
for i in smpls:
    print(len(smpls))
    
for i in smpls:
    print(len(i))
    
    
smpls = [get_features(cv2.imread("CarData/TrainImages/pos-%d.pgm" % x)) for x in range(10,20)]
smpls = [a for a in smpls if len(a) == 3200]
smpls
for i in smpls:
    print(len(i))
    
negs = [get_features(cv2.imread("CarData/TrainImages/neg-%d.pgm" % x)) for x in range(10,20)]
negs = [a for a in negs if len(a) == 3200]
for i in negs:
    print(len(i))
    
    
all_samples = np.concatenate((smpls, negs), axis=0)
len(all_samples)
all_labels = np.concatenate((np.ones(len(smpls)), np.zeros(len(negs))), axis=0)
all_labels
traindata = cv2.ml.TrainData_create(np.array(all_samples, dtype=np.float32), cv2.ml.ROW_SAMPLE, np.array(all_labels, dtype=np.float32))
get_ipython().magic(u'save 27_oct')
get_ipython().magic(u'save traindata 0-35')
