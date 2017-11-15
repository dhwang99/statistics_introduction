#encoding: utf8

import numpy as np
import pdb

'''
生成K折评测/测试集, 其k个元组，每个元组里包括留1的测试集、训练集
'''
def gen_CV_samples_by_K_folds(X, Y, K):
    CV_samples = []
    k_folds = gen_K_folds(len(X), K)
    for ki in range(K):
        ki_rst = []
        tx_pre = np.array([], dtype='int')
        tx_af = np.array([], dtype='int')
        if ki > 0:
            tx_pre = np.hstack((k_folds[:ki]))
        if ki < K-1:
            tx_af = np.hstack((k_folds[ki+1:]))
        train_xi = np.hstack((tx_pre, tx_af))
        test_xi = k_folds[ki]

        X_train_CV = X[train_xi,:]
        Y_train_CV = Y[train_xi]
        X_test_CV = X[test_xi, :]
        Y_test_CV = Y[test_xi]

        CV_samples.append((X_train_CV, Y_train_CV, X_test_CV, Y_test_CV))

    return CV_samples 

'''
生成K折集合, 其k个元组，每个元组里为均匀分布的集合 
'''
def gen_K_folds(sample_size, K):
    bsize = sample_size/K
    remain = sample_size % K
    k_folds = []

    fold_contain = np.ones(K, dtype='int') * bsize
    fold_contain[:remain] += 1
    np.random.shuffle(fold_contain)

    rids = np.arange(0, sample_size)
    np.random.shuffle(rids)
    
    beg = 0
    for interval in fold_contain:
        end = beg + interval
        #ids = np.arange(beg, end)
        ids = rids[beg:end]
        k_folds.append(ids)
        beg = end

    return k_folds

'''
基于各模型的预测值和预测方差进行一倍方差规则选取
reverse: False: 自由度从大向小排列; True: 反之
'''
def one_std_error_rule(errors, stds, reverse=False):
    min_id = np.argmin(errors)
    error_max = errors[min_id] + stds[min_id]
    if reverse == False:
        argid = np.where(errors < error_max)[0][-1]
    else:
        argid = np.where(errors < error_max)[0][0]
    
    return argid, min_id
