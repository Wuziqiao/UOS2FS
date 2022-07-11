
from scipy.sparse import coo_matrix

import sys

import warnings
warnings.filterwarnings("ignore")

import time
from functools import wraps

import numpy as np


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, ** kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running %s: %s seconds" %
              (function.__name__, str(t1 - t0))
              )
        return result

    return function_timer



class LFM():
    def __init__(self,  X_masked, K=40, lamda=0.01, alpha=0.000007, max_iter=250,discrete=0):
        '''
        Arguments
        - data: complete dataset
        - X_masked: incomplete dataset
        - K (int)       : number of latent dimensions
        - lamda (float) : regularization parameter
        - alpha (float)  : learning rate
        - max_iter:  maximum number of iterations
        - discrete: continues(0) or discrete data(1)
        '''
        # self.data = data
        self.X_masked = X_masked
        self.lamda = lamda
        self.K = K
        self.alpha = alpha
        self.max_iter = max_iter
        self.discrete = discrete

    @fn_timer
    def SGD_new(self):

        X_masked_imputzero = np.copy(self.X_masked)
        X_masked_imputzero[np.isnan(X_masked_imputzero)] = 0
        R_coo = coo_matrix(X_masked_imputzero)
        M, N = self.X_masked.shape
        self.P = np.random.rand(M, self.K)
        self.Q = np.random.rand(self.K, N)

        self.P = np.array(self.P, dtype=np.longdouble)
        self.Q = np.array(self.Q, dtype=np.longdouble)

        rmse1 = np.inf
        flag = 1

        for step in range(self.max_iter + 1):
            for ui in range(len(R_coo.data)):
                rui = R_coo.data[ui]
                u = R_coo.row[ui]
                i = R_coo.col[ui]
                if rui:
                    eui = (rui - np.dot(self.P[u, :], self.Q[:, i]))
                    self.P[u, :] = self.P[u, :] + self.alpha * 2 * (eui * self.Q[:, i] - self.lamda * self.P[u, :])
                    self.Q[:, i] = self.Q[:, i] + self.alpha * 2 * (eui * self.P[u, :] - self.lamda * self.Q[:, i])

            if not step % 5:
                rmse = self.error()
                if np.isnan(self.P).any():
                    print("NAN")
                    sys.exit(0)
                if rmse > rmse1:
                    print("  times:\t" + str(step) + '\t\t' + str(rmse1))
                    flag = 0
                    break
                rmse1 = rmse
            self.alpha = 0.9 * self.alpha
        if flag:print("  times:\t" + str(step) + '\t\t' + str(rmse1))
        return


    def error(self):
        # ratings = R.data
        # rows = R.row
        # cols = R.col
        # t0 = time.time()

        e = 0
        times = 0
        abss = 0
        preR = self.P.dot(self.Q)
        self.index = np.argwhere(~np.isnan(self.X_masked))
        for i, j in self.index:
            e = e + pow(self.X_masked[i, j] - preR[i, j], 2)
            # abss = abss + np.abs(data.at[i, j] - preR[i, j])
            times += 1
        rmse = np.sqrt(e / times)
        #print(" this time RMSE: " + str(rmse))
        # t1 = time.time()
        # print("times: ",t1-t0)
        return rmse

    def replace_nan(self):
        """
        Replace np.nan of X with the corresponding value of X_hat
        """

        X_hat = self.P.dot(self.Q)
        X = np.copy(self.X_masked)
        # if self.discrete == 1:
        #     X_hat = np.rint(X_hat)
        Nan_place = np.argwhere(np.isnan(self.X_masked))
        if self.discrete == 0:
            for i, j in Nan_place:
                X[i, j] = X_hat[i, j]
        return X

@fn_timer
def mask_types(X, p, seed):    #随机缺失
    X_masked = np.copy(X).astype(float)
    mask_indices = []
    num_rows = X_masked.shape[0]
    num_cols = X_masked.shape[1]

    for i in range(num_rows):
        np.random.seed(seed*num_rows-i) # uncertain if this is necessary
        for j in range(num_cols):
            rand_idx=np.random.choice([0,1],p = [p,1-p]) #从随机生成长度为mask_num的0，1，false不能取相同数字
            if rand_idx == 0:
                X_masked[i,j]=np.nan
    return X_masked

