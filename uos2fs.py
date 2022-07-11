import csv
import os
import numpy as np
import pandas as pd

from Analysis.analysistool import evaluationKNN, evaluationSVM, evaluationRF
from LF_Model.LFM_SGD import LFM
import matlab.engine

#Input the dataset with missing data to be trained
dataset = 'Isolet'
data = pd.read_csv('data/data_miss_/Isolet_train_1.csv',header=None)

X = np.array(data.iloc[:,:-1])
Y = np.array(data.iloc[:,[-1]])

# Build the buffer matrix
col = X.shape[1]
Buffer_Matrix_width = 15
buf_start = 0
buf_end = Buffer_Matrix_width
X_new = np.empty((X.shape[0], 0))
X_copy = np.copy(X)

#compulated the missing data with LFM
while buf_start < col:
    print("***********  buf_start:\t", str(buf_start) + '\t***********')
    if buf_end >= col:
        # This_X = X_arr[:, buf_start:]
        This_X_masked = X_copy[:, buf_start:]
    else:
        # This_X = X_arr[:, buf_start:buf_end]
        This_X_masked = X_copy[:, buf_start:buf_end]
    lfm = LFM( This_X_masked, K=40, lamda=0.1, alpha=0.001, max_iter=800)
    lfm.SGD_new()
    This_X_new = lfm.replace_nan()
    X_new = np.hstack((X_new, This_X_new))
    buf_start += Buffer_Matrix_width
    buf_end += Buffer_Matrix_width

#The completed dataset
X_new_train = np.hstack((X_new, Y))


path_LF = 'data/Fill_LF_/'
if os.path.exists(path_LF) == False:
    os.makedirs(path_LF)

savepath_train_ori = path_LF + dataset+ '_train_LF'+ '.csv'
with open(savepath_train_ori, 'w', newline='') as f:
    writer = csv.writer(f)  # 构造写入器
    for i in range(X_new_train.shape[0]):
        writer.writerow(X_new_train[i, :])


# data_train = pd.read_csv('data/Fill_LF_/Isolet_train_LF.csv',header=None)
data_train = pd.DataFrame(X_new_train)
data_test = pd.read_csv('data/data_miss_/Isolet_test_1.csv',header=None)

#Get the result of feature selection
data_fs = np.array(data_train)
data_shift = data_fs.tolist()
eng = matlab.engine.start_matlab()
data_shift = matlab.double(data_shift)
D = data_fs.shape[1]

fs_result_dep = eng.uncertain_fs(data_shift,D)
result = np.array(fs_result_dep).astype(int)
result = result[0]
print('select features: ', result)

#To verify the result of selected
result_verify = result - 1
data_train_select = np.array(data_train.iloc[:,result_verify])
train_label = np.array(data_train.iloc[:, -1])
data_test_select = np.array(data_test.iloc[:,result_verify])
test_label = np.array(data_test.iloc[:,-1])

t1 = evaluationKNN(data_train_select, train_label, data_test_select, test_label)
t2 = evaluationSVM(data_train_select, train_label, data_test_select, test_label)
t3 = evaluationRF(data_train_select, train_label, data_test_select, test_label)

print("\n")
print('The accuracy of KNN:  %.4f'%(t1))
print('The accuracy of SVM:  %.4f'%(t2))
print('The accuracy of RF:  %.4f'%(t3))
print("Mean Accuracy: %.2f\n"%((t1+t2+t3)/3))
