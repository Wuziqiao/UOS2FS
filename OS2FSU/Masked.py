import numpy as np
import pandas as pd
import os
import scipy.io as sio
import csv
from sklearn.model_selection import KFold
from LF_Model.LFM_SGD import mask_types

#Simulate data sets with missing values
if __name__ == '__main__':

    dataset = "Isolet"
    data = sio.loadmat("data/"+dataset+".mat")
    # data2 = data['data']   #COIL用data,HAPT_no_Bayes用HAPT
    # data = pd.DataFrame(data2)
    # X = np.array(data.iloc[:,:-1])
    # Y = np.array(data.iloc[:,[-1]])

    X1 = data['X']
    Y1 = data['Y']
    data = np.hstack((X1, Y1))
    X = data[:, :-1]
    Y = data[:,[-1]]

    P = [0.5]   #data missing rate
    RMSE_all = []

    for p in P:
        print("\n*******************  P:\t", str(p) + '\t***************************')
        times = 0
        RMSE_mean = []
        New_sam = KFold(n_splits=5, shuffle=True, random_state=12)

        for train_index, test_index in New_sam.split(X):

            times += 1
            X_this_time = X[train_index]
            Y_this_time = Y[train_index]
            X_test = X[test_index]
            Y_test = Y[test_index]

            X_arr = np.array(X_this_time)
            print("--------------  times:\t", str(p), '   ', str(times) + '\t-----------------------')
            X_masked = mask_types(X_this_time, p, 1)  # 随机缺失数据,p为数据缺失率
            X_masked_Y = np.hstack((X_masked,Y_this_time))
            X_test_Y = np.hstack((X_test,Y_test))

            path_LF = 'data/data_miss_/'
            if os.path.exists(path_LF) == False:
                os.makedirs(path_LF)

            savepath_masked = path_LF + dataset + '_train_' + str(times) + '.csv'
            savepath_test = path_LF + dataset + '_test_' + str(times) + '.csv'

            # 保存预测文件为CSV
            with open(savepath_masked, 'w', newline='') as f:
                writer = csv.writer(f)  # 构造写入器
                for i in range(X_masked_Y.shape[0]):
                    writer.writerow(X_masked_Y[i, :])

            with open(savepath_test, 'w', newline='') as f:
                writer = csv.writer(f)  # 构造写入器
                for i in range(X_test_Y.shape[0]):
                    writer.writerow(X_test_Y[i, :])

            break



