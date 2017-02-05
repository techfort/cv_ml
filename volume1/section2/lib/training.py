from os import path
import cv2
import numpy as np
import sift

facesdir = path.join("..","..","orl_faces")
print path.realpath(facesdir)
NUM_FACES = 40 
NUM_SAMPLES = 10
NUM_FEATURES = 25 * 128

def imgpath(facesdir, face, sample):
    return path.join(facesdir, "s%d" % face, "%d.pgm" % (sample))

traindata = []
trainlabels = []

data = [{"img": cv2.imread(path.join("..", "..", "orl_faces", "s%d" % j, "%d.pgm" % k)),
    "label": j} for j in xrange(1, NUM_FACES + 1) for k in xrange(1, NUM_SAMPLES + 1)] 

def getTrainingData():
    for s in data:
        features = sift.extract_features(s["img"])
        if len(features) ==  NUM_FEATURES:
            traindata.append(features)
            trainlabels.append(s["label"])
    TrainData = cv2.ml.TrainData_create(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))
    return {"raw": data, "traindata": TrainData}
