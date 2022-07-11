from Model.OFS.osfs import OSFS
from scipy.sparse import coo_matrix
import scipy.io as sio
from numpy.linalg import norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from Analysis.analysistool import crossValidation


#data = pd.read_csv(r"E:\pythonproject\fuzzy_feature_selection\data\wdbc.data",header=None)
data = sio.loadmat(r"E:\pythonproject\fuzzy_feature_selection\data\lungcancer.mat")
data2 = data['lungcancer']
data = pd.DataFrame(data2)

#data = pd.read_csv(r"E:\pythonproject\fuzzy_feature_selection\data\spect.csv",header=None)

#data = data.sample(frac = 1,random_state=5)

#数据预处理
# class_le = LabelEncoder()       #离散数据映射
# data[1] = class_le.fit_transform(data[1])

X = data.iloc[:,:-1]
Y = data.iloc[:,[-1]]

p = 0.8
f1 = open(r'E:\code\Basic LF\train1.txt', 'w')
f2 = open(r'E:\code\Basic LF\test1.txt', 'w')
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        n_random = np.random.rand()
        if n_random > p:
            f1.write(str(i) +"::"+str(j)+"::"+str(X.iloc[i,j]) +"\n")
        else:
            f2.write(str(i) +"::"+str(j)+"::"+str(X.iloc[i,j]) +"\n")

f1.close()
f2.close()


