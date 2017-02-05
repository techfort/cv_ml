import cv2
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
import numpy as np
from random import randint
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from plot_cm import plot_confusion_matrix

# if you have it downloaded already it will load the cached copy
mnist = fetch_mldata('MNIST original')
X = mnist.data
y = mnist.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.03, random_state=42)

# let's test an example image
test_zero = np.array(mnist.data[0]).reshape(28,28)
while True:
    cv2.imshow("zero", test_zero)
    if cv2.waitKey() == ord("q"):
        break
cv2.destroyAllWindows()

# create the KNN classifier
KNN = cv2.ml.KNearest_create()

# train the model
KNN.train(np.array(X_train, dtype=np.float32), cv2.ml.ROW_SAMPLE, np.array(y_train, dtype=np.int32))

# predictions and actual values arrays
y_pred = []

# example classifications with images
for i in range(0,20):
    rint = randint(0, 70000)
    img = mnist.data[rint]
     
    predicted = KNN.predict(np.array([mnist.data[rint]],dtype=np.float32))[1][0][0]

    cv2.imshow("Digit %d" % predicted, np.array(img).reshape(28,28))
    cv2.waitKey()
    cv2.destroyWindow("Digit %d" % predicted)

cv2.destroyAllWindows()

for i,d in enumerate(X_test): 
    print i
    predicted = KNN.predict(np.array([d],dtype=np.float32))[1][0][0]
    y_pred.append(predicted)

target_names = ["Digit %d" % x for x in np.unique(mnist.target)]

print(classification_report(y_test, y_pred, target_names=target_names))
cnf_matrix = confusion_matrix(y_test, y_pred)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=target_names,
                              title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
                              title='Normalized confusion matrix')

plt.show()
