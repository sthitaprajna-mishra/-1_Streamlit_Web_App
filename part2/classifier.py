import pickle
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

iris = datasets.load_iris()
X = iris.data
y = iris.target

clf = RandomForestClassifier()
clf.fit(X, y)

with open('clf.pkl', 'wb') as f:
	pickle.dump(clf, f)