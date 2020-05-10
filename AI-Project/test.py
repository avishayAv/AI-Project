from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


def test_accuracy(x_train, y_train, x_test, y_test):
    svm = SVC(gamma='auto', random_state=101)
    forest = RandomForestClassifier(n_estimators=10, random_state=101)
    knn = KNeighborsClassifier()
    sgd = SGDClassifier(random_state=101)
    bn = GaussianNB()

    svm.fit(x_train, y_train)
    forest.fit(x_train, y_train)
    knn.fit(x_train, y_train)
    sgd.fit(x_train, y_train)
    bn.fit(x_train, y_train)

    y1_hat = svm.predict(x_test)
    y2_hat = forest.predict(x_test)
    y3_hat = knn.predict(x_test)
    y4_hat = sgd.predict(x_test)
    y5_hat = bn.predict(x_test)

    acc1 = metrics.accuracy_score(y_test, y1_hat)
    acc2 = metrics.accuracy_score(y_test, y2_hat)
    acc3 = metrics.accuracy_score(y_test, y3_hat)
    acc4 = metrics.accuracy_score(y_test, y4_hat)
    acc5 = metrics.accuracy_score(y_test, y5_hat)

    return {'svm': round(acc1, 3), 'forest': round(acc2, 3), 'knn': round(acc3, 3), 'sgd': round(acc4, 3), 'bn': round(acc5, 3)}

