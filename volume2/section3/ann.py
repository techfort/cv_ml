import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from plot_cm import plot_confusion_matrix

target_names = [ "DAFFODIL",
        "SNOWDROP",
        "LILY VALLEY",
        "BLUEBELL"]

NUM_CLASSES = len(target_names)
NUM_SAMPLES = 80
NUM_FEATURES = 10
NUM_CH = 20 
NUM_EPOCHS = 100000

det, extr = cv2.xfeatures2d.SIFT_create(NUM_FEATURES), cv2.xfeatures2d.SIFT_create(NUM_FEATURES)
NUM_INPUTS = 2 * NUM_CH

bow_kmeans = cv2.BOWKMeansTrainer(40)
bfmatcher = cv2.BFMatcher()
bow_extractor = cv2.BOWImgDescriptorExtractor(extr, bfmatcher)

def read(img):
    return cv2.imread("../../flowers/jpg/image_%04d.jpg" % img, 0)

def compute(bow_extr, detector, img):
    return bow_extr.compute(img, detector.detect(img))

def addToBow(bow, extr, det, img):
    bow.add(extr.compute(img, det.detect(img))[1])

counter = 1
for c in range(1, NUM_CLASSES + 1):
    for s in range(1, NUM_SAMPLES + 1):
        print "processing %d" % counter
        addToBow(bow_kmeans, extr, det, read(counter))
        counter += 1

voc = bow_kmeans.cluster()
bow_extractor.setVocabulary(voc)

def ann_result(x):
    res = np.zeros((NUM_CLASSES,1), dtype=np.float32)
    res[x - 1] = 1.
    return res

def res_from_array(arr):
    return np.argmax(arr)

all_samples = []
all_labels = []
counter = 1

for c in range(1, NUM_CLASSES + 1):
    for s in range(1, NUM_SAMPLES + 1):
        print "processing %d" % counter
        all_labels.append(ann_result(c))
        all_samples.extend(compute(bow_extractor, det, read(counter))) 
        counter += 1

print len(all_samples[0])

X_train, X_test, y_train, y_test = train_test_split(all_samples, all_labels, test_size=0.15, random_state=42)

ann = cv2.ml.ANN_MLP_create()
layers = np.array([len(all_samples[0]), 10, NUM_CLASSES])
ann.setLayerSizes(layers)
ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
ann.setTermCriteria(( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1 ))
for i in range(1, NUM_EPOCHS):
    print "epoch %d" % i
    ann.train(np.array(X_train, dtype=np.float32), cv2.ml.ROW_SAMPLE, np.array(y_train, dtype=np.float32))

y_pred = []

for i, d in enumerate(X_test):
    predicted = ann.predict(np.array([d], dtype=np.float32))[0]
    y_pred.append(predicted)

Y = []
for arr in y_test:
    cls = res_from_array(arr)
    print cls
    Y.append(cls)

y_test = Y
print y_test

np.set_printoptions(precision=2)
print(classification_report(y_test, y_pred, target_names=target_names))
cnf_matrix = confusion_matrix(y_test, y_pred)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=target_names,
                              title='Confusion matrix, without normalization')

plt.show()
