#encoding: utf8

import numpy as np
import pdb
from scipy.stats import f as f_stats

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

'''
def load_data(filename='data/prostate.data.txt'):
    usecolums = range(1, 11)
    alldata = np.loadtxt(filename, skiprows=1, usecols=usecolums, converters={10:data_convert})
    std_val = alldata.std(axis=0)

    mean_data = np.copy(alldata[:, 0:8])
    mean_val = mean_data.mean(axis=0)
    mean_data = mean_data - mean_val
    std_val = mean_data.std(axis=0)
    std_data = mean_data / std_val
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
    #pdb.set_trace()
    beta_hat = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)

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
            



if __name__ == '__main__':
    X, Y, X_t, Y_t = load_data()
    beta, z_scores, test_err, pre_err = leasq(X, Y, X_t, Y_t)

    print "\n\n leasq select by zscore:"
    b2, z2, t2, p2 = leasq_select_by_zscore(X, Y, X_t, Y_t, z_scores)
    
    f_v = f_value_check(test_err, t2, len(beta), len(b2), len(X))
    print "\n\nf value check:", f_v

    print "\n\nexhaustive_select:"

    exhaustive_select(X, Y, X_t, Y_t)

