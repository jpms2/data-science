from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn import svm

clf = svm.SVC(gamma=0.001, C=100.)
digits = load_digits()
logisticRegr = LogisticRegression()
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)
clf.fit(x_train, y_train)

score = clf.score(x_test, y_test)
print(score)