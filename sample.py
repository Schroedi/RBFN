from sklearn import datasets
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from rbfn import Rbfn

iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
scaler = StandardScaler()

clf = Rbfn()
clf.fit(scaler.fit_transform(X_train), y_train)

y_pred = clf.predict(scaler.transform(X_test))
print(classification_report(y_test, y_pred))
print("\tAccuracy: %1.3f" % accuracy_score(y_test, y_pred))
