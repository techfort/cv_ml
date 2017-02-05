import cv2
import numpy as np
import os.path as path
from lib.utils import imgpath


detect = cv2.xfeatures2d.SIFT_create(20)
extract = cv2.xfeatures2d.SIFT_create()

def extract_feat(img):
    return np.array(np.ravel(extract.compute(img, detect.detect(img))[1]), dtype=np.float32)

traindir = path.join("..","..","CarData","TestImages")
images = [cv2.imread(imgpath(traindir, "test", i)) for i in xrange(1, 169)]

print len(images)

kps = [extract_feat(img) for img in images]
print "keypoints: %d" % len(kps[0])
kps = [kp for kp in kps if len(kp) == 20 * 128]


def test(svm):
    total = 0
    counter = 0
    for k in kps:
        p = svm.predict(np.array([k]))[1][0][0]
        if (p == 1.):
            total += 1.
        counter += 1.

    print "Accuracy: %f" % (total/counter * 100.0)
