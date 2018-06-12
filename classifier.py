from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB

dataset = pd.read_csv()
data = dataset.ix[:,:-1]
target = dataset.iloc[:,-1]

""" SVM """
clf = svm.SVC(gamma=0.001, C=100.)
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=0)
clf.fit(x_train, y_train)

score = clf.score(x_test, y_test)
print('SVM: ' + str(score))
predict = clf.predict(x_test)

""" K MEANS """
kmeans = KMeans(n_clusters=3)
kmeans = kmeans.fit(data)
centroids = kmeans.cluster_centers_
predict = kmeans.predict(x_test)

""" Gaussian NAIVE BAYES """
gnb = GaussianNB()
gnb.fit(x_train, y_train)
score = gnb.score(x_test, y_test)
print('Bayes: ' + str(score))
predict = clf.predict(x_test)

""" metrics 
print('Median:    ' + str(dataset['sepal_length'].median()))
print('Var:       ' + str(dataset['sepal_length'].var()))
print(dataset['sepal_length'].describe())
print(dataset.corr())"""