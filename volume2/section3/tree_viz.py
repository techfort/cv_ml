from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

dt = DecisionTreeClassifier()
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print classification_report(y_test, y_pred, target_names=iris.feature_names)
tree.export_graphviz(dt, out_file='/Users/joe/dev/python/opencv_ml/iris.dot', feature_names=iris.feature_names, class_names=iris.target_names)
