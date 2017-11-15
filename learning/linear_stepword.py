#encoding: utf8

import numpy as np
import pdb
from scipy.stats import f as f_stats
import matplotlib.pyplot as plt

from data_loader import load_data, normalize_data
from linear_regression import leasq
from cv_common import gen_CV_samples_by_K_folds, one_std_error_rule

'''
实现前后向子集选择。评价算法为 AIC, CV
'''

'''
子集选择, forward AIC
注意：要求传入数据有偏置，这样可以算出截距情况, 不需要对Y进行标准化(这个教材上的不同)
'''
def forward_subset_regression_AIC(X, Y, X_t, Y_t):
    beta_count = X.shape[1]   #包括了偏置项
    beta_id_lst = range(1, beta_count)
    subids = range(beta_count)

    last_score = 1e10
    last_subids = np.array([0])
    b, z_scores, _tre, _testmse, sigma2_hat_all, se_beta_hat = leasq(X, Y, X_t, Y_t)
    
    while len(beta_id_lst) > 0:
        min_mse = 1e10
        min_id = 0
        _subs = np.hstack((last_subids, np.array([0])))
        min_subs = np.copy(_subs)
        for id in range(len(beta_id_lst)):
            _subs[-1] = beta_id_lst[id]
            _X = X[:, _subs]
            _X_t = X_t[:, _subs]
            beta, z_scores, _tre, _testmse, sig2_hat, se_beta_hat = leasq(_X, Y, _X_t, Y_t)
            if _tre < min_mse:
                min_mse = _tre
                min_id = id 
                min_subs = np.copy(_subs)
        
        min_score = min_mse*len(Y)/sigma2_hat_all + 2*(len(min_subs)-1)
        print "subset is: %s, AIC: %.4f" % (min_subs, min_score)
        if last_score > min_score:
           last_score = min_score
           beta_id_lst.pop(min_id)
           last_subids = np.copy(min_subs)
        else:
           break

    print "best subset is: %s, AIC: %.4f" % (last_subids, last_score)

'''
子集选择, backward AIC
注意：要求传入数据有偏置，这样可以算出截距情况, 不需要对Y进行标准化(这个教材上的不同)
'''
def backward_subset_regression_AIC(X, Y, X_t, Y_t):
    beta_count = X.shape[1]   #包括了偏置项
    beta_id_lst = range(1, beta_count)
    subids = range(beta_count)

    last_score = 1e10
    last_subids = np.array([0])
    b, z_scores, _tre, _testmse, sigma2_hat_all, se_beta_hat = leasq(X, Y, X_t, Y_t)
    last_subids = np.array(beta_id_lst,dtype='int')
    
    while len(beta_id_lst) > 0:
        min_mse = 1e10
        min_id = 0
        min_subs = None  
        for id in range(len(beta_id_lst)):
            cur_beta = list(beta_id_lst)
            cur_beta.pop(id)
            _subs = np.hstack((np.array([0], dtype='int'), np.array(cur_beta, dtype='int')))
            _X = X[:, _subs]
            _X_t = X_t[:, _subs]
            beta, z_scores, _tre, _testmse, sig2_hat, se_beta_hat = leasq(_X, Y, _X_t, Y_t)
            if _tre < min_mse:
                min_mse = _tre
                min_id = id 
                min_subs = np.copy(_subs)
        min_score = min_mse*len(Y)/sigma2_hat_all + 2*(len(min_subs)-1)
        print "subset is: %s, AIC: %.4f" % (min_subs, min_score)
        if last_score > min_score:
           last_score = min_score
           beta_id_lst.pop(min_id)
           last_subids = np.copy(min_subs)
        else:
           break

    print "best subset is: %s, AIC: %.4f" % (last_subids, last_score)


'''
前后向子集选择，by CV
不能实现

CV更多用于 lasso, ridge等等回归、分类中进行模型选择
'''

if __name__ == '__main__':
    #读取数据，并进行标准化
    X, Y, X_t, Y_t, dt_cf = load_data()
    print "forward_subset_regression_AIC:"
    forward_subset_regression_AIC(X, Y, X_t, Y_t)

    print ""
    print "backward_subset_regression_AIC:"
    backward_subset_regression_AIC(X, Y, X_t, Y_t)
