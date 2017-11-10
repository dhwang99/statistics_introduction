#encoding: utf8

import numpy as np
import pdb
from scipy.stats import f as f_stats
import matplotlib.pyplot as plt

'''
load data from file
return:
    train src data, test src data, normed training src dtaa, normed test src data
'''
def data_convert(val):
    if val == 'T':
        return 2
    elif val == 'F':
        return 1
    return 0

'''
问题1：
1. 关于标准化，是每一个因子在全集上标准化，还只是在测试集上标标准化？
我用的是在全集上标准化，这个应该更合理一些
全集上标准化

问题2：
2. 在求协方差时，np.cov的算法：sum(Xi-X_mean)(Yi-Y_mean)/(len(samples)-1)
   在实际上我算的时候，并没有减1. 这两个是不是第1个更准(无偏)但实际上用的时候是第2个?
   mean_data.T * mean_data / n

type:
    0:原始数据
    1:均值化
    2:标准化
need_bias: 是否需要偏置项(1)
'''
def load_data(filename='data/prostate.data.txt', type=2, need_bias=1):
    usecolums = range(1, 11)
    alldata = np.loadtxt(filename, skiprows=1, usecols=usecolums, converters={10:data_convert})
    std_val = alldata.std(axis=0)

    std_data = np.copy(alldata[:, 0:8])
    if type > 0:
        mean_val = std_data.mean(axis=0)
        std_data = std_data - mean_val
        if type == 2:
            std_val = std_data.std(axis=0)
            std_data = std_data / std_val

    if need_bias > 0:
        std_data = np.hstack((np.ones((len(std_data), 1)), std_data))

    t_indicate = np.where(alldata[:,9] == 2)
    f_indicate = np.where(alldata[:,9] == 1)

    X_train = np.take(std_data, t_indicate[0], axis=0)
    Y_train = np.take(alldata[:,8], t_indicate[0], axis=0)

    X_test = np.take(std_data, f_indicate[0], axis=0)
    Y_test = np.take(alldata[:,8], f_indicate[0], axis=0)

    return X_train, Y_train, X_test, Y_test


def leasq(X_train, Y_train, X_test, Y_test):
    X = X_train
    Y = Y_train

    #X.T(Y - X*beta_hat) = 0
    # beta_hat = inv(X.T*X)*X.T*Y
    #预估beta
    # X = QR
    # R * beta = R.inv * Q.T * Y, R为上三解矩阵
    # beta_hat = R.inv * Q.T * Y
    #pdb.set_trace()
    #beta_hat = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
    Q,R = np.linalg.qr(X)
    beta_hat = np.dot(np.dot(np.linalg.inv(R), Q.T), Y)

    print "Beta hat: %s" % beta_hat
    print ""

    #计算训练误差
    test_err = np.dot(X, beta_hat.T) - Y
    test_err = np.dot(test_err, test_err)/len(Y)
    print "test squre error:", test_err

    #计算beta的样本协方差
    sigma2_hat = np.dot(test_err, test_err) / (len(X) - X.shape[1] - 1)
    convs = np.linalg.inv(np.dot(X.T, X)) * sigma2_hat

    se_hat = map(lambda i:np.sqrt(convs[i,i]), range(len(beta_hat)))
    print "Se hat: %s" % se_hat
    print ""

    #计算Z_score: Z_alpha = beta_hat / convs[i:i]
    z_scores = map(lambda i:beta_hat[i]/se_hat[i], range(len(beta_hat)))
    fs = "Z_scores:" + ", ".join(["%.4f"] * len(z_scores))
    print fs % tuple(z_scores)
    print ""

    #计算预测误差 
    pre_err = np.dot(X_test, beta_hat) - Y_test
    pre_err = np.dot(pre_err, pre_err) /len(Y_test)
    print "predict mean error: %.4f" % pre_err
    return beta_hat, z_scores, test_err, pre_err


'''
用zscore进行特征选择
'''
def leasq_select_by_zscore(X, Y, X_t, Y_t, z_scores):
    #feature index
    z_scores = np.array(z_scores)
    f_i = np.where(z_scores > 1.95)[0]
    X = X[:, f_i]
    X_t = X_t[:, f_i]
    
    print "features: %s" % f_i
    return leasq(X, Y, X_t, Y_t)

'''
用F值进行系数显著性检查
F = [(RSS1-RSS0)/(p1-p0)] / [RSS1/(n-p1-1)]
F ~ F(p1-p0, n-p1-1)
n比较大时，F ~ chi2(p1 - p0)
RSS1为较大模型的rss
'''
def f_value_check(rss1, rss0, fi1, fi0, n):
    df0 = fi1 - fi0
    df1 = n - fi1 - 1
    F = np.abs(rss1 - rss0) * df0 / (df1 * rss1)
    ppf = f_stats.ppf(F, df0, df1)

    return ppf



'''
子集选择
穷举法找最优模型 
'''
import itertools
def exhaustive_select(X, Y, X_t, Y_t):
    beta_count = X.shape[1]
    beta_id_lst = range(1, beta_count) 
    mse_lst = []
    for sub_count in range(0, beta_count):
        sub_list = list(itertools.combinations(beta_id_lst, sub_count))
        mse_train = 1e10
        mse_test = 1e10
        test_sub = []
        for sub in sub_list:
            sub = np.hstack((np.array([0]), np.array(sub, dtype='int')))
            _X = X[:, sub]
            _X_t = X_t[:, sub]
    
            beta, z_scores, train_err, test_err = leasq(_X, Y, _X_t, Y_t)
            if mse_test > test_err:
                mse_train = train_err
                mse_test = test_err
                test_sub = sub

        mse_lst.append((sub_count, test_sub, mse_train, mse_test))
    
    print mse_lst

    return

'''
生成K折集合
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
'''
def one_std_error_rule(errors, stds):
    min_id = np.argmin(errors)
    error_max = errors[min_id] + stds[min_id]
    argid = np.where(errors < error_max)[0][-1]
    
    return argid, min_id


'''
在特征之间有一定线性相关度时（协方差不为0），一个大的特征参数有可能会增加而另一些变为负数或趋于0
这导致方差很大
L2通过收缩参数，减小了方差的影响
Y = beta * X + lambda * sum(beta^2)
 beta_hat = inv(X.T*X -lamb*I)*X.T*Y
用svd分解求解
 X = UDV.T
 beta_hat = V*inv(D*D + lamb*I)*D*U.T * y
 df(lamb) = sum [dj**2/(dj**2 + lamb)]

 用k折交叉验证法进行df选择
 注意下：
 1. 如果不有bias/intercept, 对Y要进行均值化
 2. numpy 的 svd, 返回的是V的转秩，用的时候需要再转秩一下
 3. 求估计(均值)的标准差时，记得除以sqrt(n) 
'''
def leasq_with_L2_new():
    X_train, Y_train, X_test, Y_test = load_data(type=0, need_bias=0)

    X_mean = X_train.mean(axis=0)
    X_train = X_train - X_mean
    X_std = X_train.std(axis=0)
    X_train /= X_std

    Y_mean = Y_train.mean()
    Y_train = Y_train - Y_mean

    K = 10
    lamb_lst = np.logspace(-5, 3, 30)
    train_mid_rst = []
    k_folds = gen_K_folds(len(X_train), K)

    for ki in range(K):
        ki_rst = []
        tx_pre = np.array([], dtype='int')
        tx_af = np.array([], dtype='int')
        if ki > 0:
            tx_pre = np.hstack((k_folds[:ki]))
        if ki < K-1:
            tx_af = np.hstack((k_folds[ki+1:]))
        train_xi = np.hstack((tx_pre, tx_af))

        X_train_CV = X_train[train_xi,:]
        Y_train_CV = Y_train[train_xi]
        X_test_CV = X_train[k_folds[ki], :]
        Y_test_CV = Y_train[k_folds[ki]]

        X = X_train_CV
        Y = Y_train_CV
        U,D,V = np.linalg.svd(X)
        V = V.T
        D2 = D**2
        D_UT_y = np.dot(np.dot(np.diag(D), U.T[:len(D2),:]), Y)
        I_b = np.eye(X.shape[1])

        for lamb in lamb_lst:
            #计算beta估计
            '''
            开始时算得有问题，原因:
            svd分解时，返回的是VT, 不是V, 用的时候需要注意一下.fix这个bug后，beta_hat2, beta_hat4, beta_hat基本一致了
            beta_hat2 = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + I_b * lamb), X.T), Y)
            beta_hat4 = np.dot(np.dot(V, np.linalg.inv(np.diag(D2) + I_b * lamb)), D_UT_y)
            beta_hat5 = np.dot(np.dot(V, np.diag(np.diag(1/(np.diag(D2) + I_b * lamb)))), D_UT_y)
            '''
            beta_hat = np.dot(np.dot(V, np.diag(1/(D2 + lamb))), D_UT_y)
    
            #计算训练误差
            train_err = np.dot(X, beta_hat.T) - Y
            train_rss = np.dot(train_err, train_err)/len(Y)
    
            #计算预测误差 
            test_err = np.dot(X_test_CV, beta_hat) - Y_test_CV
            test_rss = np.dot(test_err, test_err) /len(Y_test_CV)
    
            #计算自由度
            df = np.sum(D2/(D2+lamb))
    
            ki_rst.append((lamb, beta_hat, df, train_rss, test_rss))
        
        train_mid_rst.append(ki_rst)

    #计算不同lamb下的训练误差和方差
    dfs = np.zeros(len(lamb_lst))
    rss_means = np.zeros(len(lamb_lst))
    rss_mean_stds = np.zeros(len(lamb_lst))

    for lid in range(len(lamb_lst)):
        #lambda[lid]下的df
        dfs[lid] = train_mid_rst[0][lid][2]
        #K折CV下误差均值和标准差
        test_rsses = np.array(map(lambda i:train_mid_rst[i][lid][4], range(0,K)))
        rss_means[lid] = test_rsses.mean()
        #注意：这儿求的是估计的标准差 1/K mean(sum(Xi)),  故而要除以K
        rss_mean_stds[lid] = test_rsses.std()/np.sqrt(K)

    best_lamb_id, minid = one_std_error_rule(rss_means, rss_mean_stds)
    print "Best lambid: %d, lambda: %.4f, degree of free: %.4f" % (best_lamb_id, lamb_lst[best_lamb_id], dfs[best_lamb_id]) 
    
    one_std_val = rss_means[minid] + rss_mean_stds[minid]
    plt.plot((dfs[0],dfs[-1]), (one_std_val, one_std_val), 'r-')
    plt.errorbar(dfs, rss_means, yerr=rss_mean_stds, fmt='-o')
    plt.savefig('images/rss_errorbar.png', format='png')

    #用K折选出来的最优lambda进行回归预测




if __name__ == '__main__':
    X, Y, X_t, Y_t = load_data()
    '''
    beta, z_scores, test_err, pre_err = leasq(X, Y, X_t, Y_t)

    print "\n\n leasq select by zscore:"
    b2, z2, t2, p2 = leasq_select_by_zscore(X, Y, X_t, Y_t, z_scores)
    
    f_v = f_value_check(test_err, t2, len(beta), len(b2), len(X))
    print "\n\nf value check:", f_v

    print "\n\nexhaustive_select:"
    exhaustive_select(X, Y, X_t, Y_t)
    '''

    print "\n\nleasq by L2:"
    leasq_with_L2_new()
