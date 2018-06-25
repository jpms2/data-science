from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm, tree
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


dataset = pd.read_csv("C://Users//jpms2//Desktop//data-science//data-science//ashley_madison_preprocessed.csv")

def check_married(column):
    result = []
    for row in column:
        if('Attached' in row):
            result.append(True)
        else:
            result.append(False)
    return result

def is_woman(column):
    result = []
    for row in column:
        if('Female' in row):
            result.append(True)
        else:
            result.append(False)
    return result


dataset['is_married'] = np.where(check_married(dataset['profile_relationship_desc']), 1, 0)
dataset = pd.concat([dataset, pd.get_dummies(dataset.iloc[:,11])], axis=1)
dataset = pd.concat([dataset, pd.get_dummies(dataset.iloc[:,15])], axis=1)
dataset['is_woman'] = np.where(is_woman(dataset['gender_desc']), 1, 0)
#Hipotese 1: Calcular se um usuário bebe ou não baseado em suas preferências sexuais
#data = dataset.iloc[:, 17:-1]
#target = dataset.iloc[:, 8]
#SVM: 0.89 (+/- 0.01), Decision Tree: 0.887050430992, Bayes: 0.753987912415

#Hipotese 2: Calcular se o usuario é casado a partir das preferencias
#data = dataset.iloc[:,17:-1]
#target = dataset.iloc[:,162]
#SVM: 0.70 (+/- 0.28), Decision Tree: 0.762187871581, Bayes: 0.63991280222

#Hipotese 3: Calcular se o usuario tem preferencia por encontros únicos a partir de se ele é casado, se bebe, pela idade e etnia
#data = dataset.iloc[:, [8, 16, 162,163,164,165,166,167,168,169,170,171]]
#target = dataset.iloc[:,24]
#SVM: 0.59 (+/- 0.10), Decision Tree: 0.614447086801, Bayes: 0.570947284978

#Hipotese 4: Homens participantes são mais propensos a serem obesos
#dset = dataset.loc[dataset['is_woman'] == 1]
#cluster = dset.iloc[:, [6,16]]

#Hipotese 5: Calcular o porte físico a partir de peso, altura, etnia e estado civil
data = dataset.iloc[:,[6,7,162,163,164,165,166,167,168,169,170,171,162]]
target = dataset.iloc[:,12]



# SVM 
clf = svm.SVC(gamma=0.001, C=100.)

#k-fold
scores = cross_val_score(clf, data, target, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#predict = clf.predict(x_test)

#Decision tree
X_train, X_test, y_train, y_test = train_test_split( data, target, test_size = 0.3, random_state = 100)
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)
score = clf_entropy.score(X_test, y_test)
print('Decision Tree: ' + str(score))
predict = confusion_matrix(X_train, clf_entropy.predict(X_test))
print(predict)

# Gaussian NAIVE BAYES
gnb = GaussianNB()
gnb.fit(X_train, y_train)
score = gnb.score(X_test, y_test)
print('Bayes: ' + str(score))


"""
# K MEANS 
X = cluster.values

wcss = []
# Elbow method for k-means
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'random')
    kmeans.fit(X)
    print(i,kmeans.inertia_)
    wcss.append(kmeans.inertia_)  
plt.plot(range(1, 11), wcss)
plt.title('O Metodo Elbow')
plt.xlabel('Numero de Clusters')
plt.ylabel('WSS') #within cluster sum of squares
plt.show()

kmeans = KMeans(n_clusters = 3, init = 'random')
kmeans.fit(X)

plt.scatter(X[:, 0], X[:,1], s = 100, c = kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'red',label = 'Centroids')
plt.show()
"""

""" metrics 
print('Median:    ' + str(dataset['sepal_length'].median()))
print('Var:       ' + str(dataset['sepal_length'].var()))
print(dataset['sepal_length'].describe())
print(dataset.corr())
"""