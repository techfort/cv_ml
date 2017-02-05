import cv2
import numpy as np
import os.path as path
from lib.utils import imgpath
import carsvm

traindir = path.join("..","..","CarData","TrainImages")
print path.realpath(traindir)

POS = 549
NEG = 499
NUM_FEATURES = 20
LEN_FEATURES = NUM_FEATURES * 128

detect = cv2.xfeatures2d.SIFT_create(20)
extract = cv2.xfeatures2d.SIFT_create()

# def imgpath(basepath, klass, number):
#     return path.join(basepath, "%s-%d.pgm" % (klass, number))

def extract_features(klass, image):
    img = cv2.imread(imgpath(traindir, klass, image))
    return np.array(np.ravel(extract.compute(img, detect.detect(img))[1])[0 : LEN_FEATURES], dtype=np.float32)

positives = [extract_features("pos", img) for img in xrange(1, POS + 1)]
negatives = [extract_features("neg", img) for img in xrange(1, NEG + 1)]

positives = [p for p in positives if len(p) == LEN_FEATURES]
negatives = [n for n in negatives if len(n) == LEN_FEATURES]

trainimages = positives + negatives
trainlabels = np.append(np.ones(len(positives)), np.zeros(len(negatives))) 

print "Img: %d, responses: %d" % (len(trainimages), len(trainlabels))

traindata = cv2.ml.TrainData_create(np.array(trainimages), cv2.ml.ROW_SAMPLE, np.array(trainlabels, dtype=np.int32))

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setGamma(0.4)
svm.setC(30)
svm.setTermCriteria(criteria)

svm.train(traindata)

svm.save("carsvm.data")
carsvm.test(svm)
