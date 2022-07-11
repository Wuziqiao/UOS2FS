import numpy as np
from sklearn.model_selection import KFold
from numpy.linalg import norm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")

def accuracy(test_labels, pred_lables):
    correct = np.sum(test_labels == pred_lables)
    n = len(test_labels)
    return float(correct) / n *100

def evaluationKNN(train_data,train_label, test_data,test_label,k= 5):
    knn = KNeighborsClassifier(n_neighbors=k)
    result_set = (knn.fit(train_data, train_label).predict(test_data))
    test_labels = test_label.flatten()
    score = accuracy(test_labels, result_set)
    # print('KNN Accuracy: ' + repr(score) + '%')
    return score

def evaluationSVM(train_data,train_label, test_data,test_label):
    clf = svm.SVC()
    clf.fit(train_data, train_label.ravel())
    result = clf.predict(test_data)
    score = accuracy(result, test_label.flatten())
    # print('SVM Accuracy: ' + repr(score) + '%')
    return score

def evaluationRF(train_data,train_label, test_data,test_label):
    rf = RandomForestClassifier(n_estimators=80, oob_score=True, random_state=20)
    rf.fit(train_data,train_label.flatten())
    result_rf = rf.predict(test_data)
    score = accuracy(result_rf, test_label.flatten())
    # print('RandomForest Accuracy: ' + repr(score) + '%')
    return score

def crossValidationRF(X,Y):
    New_sam = KFold(n_splits=5, shuffle=True,random_state=12)
    scores = []
    for train_index, test_index in New_sam.split(X):
        rf = RandomForestClassifier(n_estimators=80,oob_score= True,random_state=20)
        rf.fit(X[train_index],Y[train_index].flatten())
        result_rf = rf.predict(X[test_index])
        score = accuracy(result_rf, Y[test_index].flatten())
        scores.append(score)
        print('RandomForest Accuracy: ' + repr(score) + '%')
        scores.append(score)
    print('accuracy: %.2f +/- %.2f\n' % (np.mean(scores), np.std(scores)))
    return np.mean(scores)


def crossValidationSVM(X,Y):
    New_sam = KFold(n_splits=5, shuffle=True,random_state=12)
    scores = []
    for train_index, test_index in New_sam.split(X):
        clf = svm.SVC()
        clf.fit(X[train_index], Y[train_index].ravel())
        # clf = svm.SVR(kernel='rbf',degree = 3,gamma='scale',C=1.0)
        # clf.fit(X[train_index], Y[train_index].ravel())
        result = clf.predict(X[test_index])
        score = accuracy(result,Y[test_index].flatten())
        scores.append(score)
        print('SVM Accuracy: ' + repr(score) + '%')
        scores.append(score)
    print('accuracy: %.2f +/- %.2f\n' % (np.mean(scores), np.std(scores)))
    return np.mean(scores)

def crossValidationKNN(features,labels):
    # New_sam = KFold(n_splits=5, random_state=50, shuffle=True)
    # scores = []
    # for train_index, test_index in New_sam.split(dataset):  # 对数据建立k折交叉验证的划分
        # for test_index,train_index in New_sam.split(Sam):  # 默认第一个参数是训练集，第二个参数是测试集
        # print(train_index,test_index)
    scores = []
    knn = KNeighborsClassifier(n_neighbors=5)
    kf = KFold(n_splits=5,shuffle=True,random_state=12)
    for train_index, test_index in kf.split(features):
        result_set = (knn.fit(features[train_index], labels[train_index]).predict(features[test_index]))
        test_labels = labels[test_index].flatten()
        score = accuracy(test_labels,result_set)
        print('KNN Accuracy: ' + repr(score) + '%')
        scores.append(score)
    print('accuracy: %.2f +/- %.2f\n' % (np.mean(scores), np.std(scores)))
    return np.mean(scores)




def cal_cov_and_avg(samples):
    """
    给定一个类别的数据，计算协方差矩阵和平均向量
    :param samples:
    :return:
    """
    u1 = np.mean(samples, axis=0)
    cov_m = np.zeros((samples.shape[1], samples.shape[1]))
    for s in samples:
        t = s - u1
        cov_m += t * t.reshape(2, 1)
    return cov_m, u1


def fisher(c_1, c_2):
    """
    fisher检验算法实现
    :param c_1:
    :param c_2:
    :return:
    """
    cov_1, u1 = cal_cov_and_avg(c_1)
    cov_2, u2 = cal_cov_and_avg(c_2)
    s_w = cov_1 + cov_2
    u, s, v = np.linalg.svd(s_w)  # 奇异值分解
    s_w_inv = np.dot(np.dot(v.T, np.linalg.inv(np.diag(s))), u.T)
    return np.dot(s_w_inv, u1 - u2)

