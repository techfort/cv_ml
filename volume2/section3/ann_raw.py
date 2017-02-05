import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from plot_cm import plot_confusion_matrix
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from imutils import paths
import argparse
import matplotlib.pyplot as plt

RESIZE = 32

def img2vec(img, size=(RESIZE, RESIZE)):
    return cv2.resize(img, size).flatten()

parser = argparse.ArgumentParser()
parser.add_argument("-e", type=int, default=100, help="number of epochs")
parser.add_argument("-hid", type=int, default=8, help="number of nodes in hidden layer")
args = parser.parse_args()

target_names = ["daffodil", "snowdrop", "lily valley", "bluebell"]
NUM_CLASSES = len(target_names)
NUM_SAMPLES = 80
TOTAL_IMAGES = NUM_SAMPLES * NUM_CLASSES
EPOCHS = args.e
VEC_LENGTH = RESIZE * RESIZE * 3
image_paths = list(paths.list_images("../../flowers/jpg"))
NUM_HIDDEN = args.hid

counter = 1
data, labels = [], []

for i, image_path in enumerate(image_paths):
    image = cv2.imread(image_path)
    mask = cv2.inRange(image, np.array([0, 100, 0], dtype=np.uint8), np.array([20, 250, 50], dtype=np.uint8))
    image = cv2.bitwise_not(image, image, mask = mask)
    label = i / NUM_SAMPLES
    vec = img2vec(image)
    if len(vec) == VEC_LENGTH:
        data.append(vec)
        print "label %d" % label
        labels.append(target_names[label])
        print "processed %d" % counter
        counter += 1
    else:
        print "skipping vector of length %d" % len(vec)
    if counter > TOTAL_IMAGES:
        break

data = np.array(data) / 255.0
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels)
print "labels, data: %d, %d" % (len(labels), len(data))
print "sample len: %d" % len(data[0])
LAYERS = np.array([VEC_LENGTH, NUM_HIDDEN, NUM_CLASSES])

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=3)

ann = cv2.ml.ANN_MLP_create()
ann.setLayerSizes(LAYERS)
ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)

scores = []
traindata = cv2.ml.TrainData_create(np.array(X_train, dtype=np.float32), cv2.ml.ROW_SAMPLE, np.array(y_train, dtype=np.float32))

ann.train(traindata, cv2.ml.ANN_MLP_NO_OUTPUT_SCALE | cv2.ml.ANN_MLP_NO_INPUT_SCALE)
for i in range(1, EPOCHS):
    print "epoch %d" % i
    ann.train(traindata, cv2.ml.ANN_MLP_UPDATE_WEIGHTS | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE | cv2.ml.ANN_MLP_NO_INPUT_SCALE)
    Y = []
    for d in X_test:
        predicted = ann.predict(np.array([d], dtype=np.float32))[0]
        Y.append(predicted)
    Y = np_utils.to_categorical(np.array(Y, dtype=np.int32), 4)
    try:
        scores.append(accuracy_score(y_test, Y))
    except ValueError as e:
        print Y
        print str(e)
        exit()

print classification_report(y_test, Y, target_names=target_names)
plt.plot(range(1, len(scores) + 1), scores, label="scores")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

print np.argmax(scores)
