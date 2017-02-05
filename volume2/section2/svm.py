import cv2
import numpy as np
import os.path as path
import argparse as arg

parser = arg.ArgumentParser(description="SVM demo")
parser.add_argument("-numf", type=int, help="please specify a number of integers")
args = parser.parse_args()

print("Creating SVM with %d number of features per sample" % args.numf)

NUM_FEATURES = args.numf
FEATURE_LENGTH = NUM_FEATURES * 128

def extract(img, detector, extractor):
    return np.array(np.ravel(extractor.compute(img, detector.detect(img))[1])[0:FEATURE_LENGTH])

def get_extractor(detector, extractor):
    def extr(clazz, img):
        return extract(get_img(clazz, img), detector, extractor)
    return extr

def get_img(clazz, img):
    return cv2.imread(path.join("..","..","CarData","TrainImages","%s-%d.pgm" % (clazz, img)))

def SVM(C):
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(C)
    return svm

def accuracy(tp, tn, fp, fn):
    return float(tp + tn) / float(tp + tn + fp + fn)

det, ext = cv2.xfeatures2d.SIFT_create(NUM_FEATURES), cv2.xfeatures2d.SIFT_create(NUM_FEATURES)

extract_features = get_extractor(det, ext)

positives = [extract_features("pos",x) for x in range(0, 300)]
negatives = [extract_features("neg",x) for x in range(0, 300)]

positives = [x for x in positives if len(x) == FEATURE_LENGTH]
negatives = [x for x in negatives if len(x) == FEATURE_LENGTH]

all_samples = np.concatenate((positives, negatives), axis = 0)
all_labels = np.concatenate((np.array(np.ones(len(positives)), dtype=np.int32), np.array(np.zeros(len(negatives)), dtype=np.int32)), axis=0)

traindata = cv2.ml.TrainData_create(np.array(all_samples, dtype=np.float32), cv2.ml.ROW_SAMPLE, np.array(all_labels))

C = 1.

def main():
    print("Creating SVM with C=%f" % C)
    svm = SVM(C)

    svm.train(traindata)

    positive_test = [extract_features("pos",x) for x in range(0, 550)]
    negative_test = [extract_features("neg",x) for x in range(0, 480)]
    
    positive_test = [x for x in positive_test if len(x) == FEATURE_LENGTH]
    negative_test = [x for x in negative_test if len(x) == FEATURE_LENGTH]
    print("Predicting positive samples")
    pos_pred = svm.predict(np.array(positive_test))[1]
    unique, counts = np.unique(pos_pred, return_counts=True)
    res = dict(zip(unique, counts))
    tp = res[1.]
    print("TP for SVM(C=%f) is %d" % (svm.getC(), tp))
    fn = len(positive_test) - tp

    neg_pred = svm.predict(np.array(negative_test))[1]
    nunique, ncounts = np.unique(neg_pred, return_counts=True)
    nres = dict(zip(nunique, ncounts))
    tn = 0
    try:
        tn = nres[0.]
    except:
        tn = 0
    print("TN for SVM(C=%f) is %d" % (svm.getC(), tn))
    fp = len(negative_test) - tn
    print("Accuracy for SVM(C=%f) is %f" % (svm.getC(), accuracy(tp, tn, fp, fn))) 
    
main()
