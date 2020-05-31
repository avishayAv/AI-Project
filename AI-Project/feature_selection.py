import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import preprocessing
from sklearn.model_selection import KFold
from test import *
import collections,functools,operator

def selectKBest_features_selection(x: pd.DataFrame, y: pd.DataFrame, k):
    selector = SelectKBest(mutual_info_regression, k=k)
    selector.fit(x, y)
    features = selector.get_support(indices=True)
    return x.columns.values[features]

def findKBest_features_selection(x:pd.DataFrame,y:pd.DataFrame,num_features):
    split = 5
    kf = KFold(n_splits=split)
    max_svm_features = []
    max_svm= 0
    max_forest_features = []
    max_forest = 0
    max_knn_features = []
    max_knn= 0
    for i in range(1,num_features):
        print(i)
        features = selectKBest_features_selection(x, y, i)
        tmp =[]
        for k, (train_index, test_index) in enumerate(kf.split(x)):
           tmp.append(test_accuracy(x[features].iloc[list(train_index)], y.iloc[list(train_index)],
                                    x[features].iloc[list(test_index)], y.iloc[list(test_index)]))
        result = dict(functools.reduce(operator.add,map(collections.Counter,tmp)))
        for key in result:
            result[key] = result[key]/split
        if result['svm'] > max_svm:
            max_svm = result['svm']
            max_svm_features = features
        if result['forest'] > max_forest:
            max_forest = result['forest']
            max_forest_features = features
        if result['knn'] > max_knn:
            max_knn = result['knn']
            max_knn_features = features
    print("max_svm = {} ,max_forest = {} , max_knn = {}".format(max_svm,max_forest,max_knn))
    print("svm_featues = {}, size = {}".format(max_svm_features,len(max_svm_features)))
    print("forest_featues = {}, size = {}".format(max_forest_features,len(max_forest_features)))
    print("knn_featues = {}, size = {}".format(max_knn_features,len(max_knn_features)))




def ExtraTree_feature_selection(data_X, data_Y, k):
    clf = ExtraTreesRegressor(n_estimators=50)
    forest = clf.fit(data_X, data_Y)
    X = data_X.values
    y = data_Y.values
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    #print("Feature ranking:")

    #for f in range(X.shape[1]):
        #print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()
    return indices[:k]

def findKExtraTree_feature_selection(x, y, num_features):
    split = 5
    kf = KFold(n_splits=split)
    max_svm_features = []
    max_svm = 0
    max_forest_features = []
    max_forest = 0
    max_knn_features = []
    max_knn = 0
    for i in range(1, num_features):
        print(i)
        features_indices = ExtraTree_feature_selection(x, y, i)
        features = list(x.columns[features_indices])
        tmp = []
        for k, (train_index, test_index) in enumerate(kf.split(x)):
            tmp.append(test_accuracy(x[features].iloc[list(train_index)], y.iloc[list(train_index)],
                                     x[features].iloc[list(test_index)], y.iloc[list(test_index)]))
        result = dict(functools.reduce(operator.add, map(collections.Counter, tmp)))
        for key in result:
            result[key] = result[key] / split
        if result['svm'] > max_svm:
            max_svm = result['svm']
            max_svm_features = features
        if result['forest'] > max_forest:
            max_forest = result['forest']
            max_forest_features = features
        if result['knn'] > max_knn:
            max_knn = result['knn']
            max_knn_features = features
    print("max_svm = {} ,max_forest = {} , max_knn = {}".format(max_svm, max_forest, max_knn))
    print("svm_featues = {}, size = {}".format(max_svm_features, len(max_svm_features)))
    print("forest_featues = {}, size = {}".format(max_forest_features, len(max_forest_features)))
    print("knn_featues = {}, size = {}".format(max_knn_features, len(max_knn_features)))


