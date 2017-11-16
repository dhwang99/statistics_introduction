#encoding: utf8

import numpy as np
import pdb
from scipy.stats import f as f_stats
import matplotlib.pyplot as plt

from data_loader import load_data

'''
sub: 使用的特征列编号
'''
def leasq(X_train, Y_train, X_test, Y_test, sub=None):
    '''
    X = X_train
    Y = Y_train
    X.T(Y - X*beta_hat) = 0
    beta_hat = inv(X.T*X)*X.T*Y
    #预估beta
    X = QR
    R * beta = R.inv * Q.T * Y, R为上三解矩阵
    beta_hat = R.inv * Q.T * Y
    beta_hat = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
    '''

    Q,R = np.linalg.qr(X_train)
    beta_hat = np.dot(np.dot(np.linalg.inv(R), Q.T), Y_train)

    #计算训练误差
    train_l1 = np.dot(X_train, beta_hat.T) - Y_train
    train_rss = np.dot(train_l1, train_l1)  
    train_mse = train_rss/len(Y_train)

    #计算beta的样本协方差
    sigma2_hat = train_rss / (len(Y_train) - X_train.shape[1] - 1)
    convs = np.linalg.inv(np.dot(X_train.T, X_train)) * sigma2_hat

    se_beta_hat = map(lambda i:np.sqrt(convs[i,i]), range(len(beta_hat)))

    #计算Z_score: Z_alpha = beta_hat / convs[i:i]
    z_scores = map(lambda i:beta_hat[i]/se_beta_hat[i], range(len(beta_hat)))

    #计算预测误差 
    test_l1 = (np.dot(X_test, beta_hat) - Y_test)
    test_rss = np.dot(test_l1, test_l1)
    #test_mse = np.dot(test_mse, test_mse) /len(Y_test)
    test_mse = test_rss/len(Y_test)
    return beta_hat, z_scores, train_mse, test_mse, sigma2_hat, se_beta_hat


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


if __name__ == '__main__':
    X, Y, X_t, Y_t,dt = load_data()
    beta, z_scores, train_mse, test_mse, sig2_hat, se_beta_hat = leasq(X, Y, X_t, Y_t)

    print "Beta hat: %s" % beta
    print ""
    fs = "Z_scores:" + ", ".join(["%.4f"] * len(z_scores))
    print fs % tuple(z_scores)
    print ""
    print "train mean square error:", train_mse
    print "test mean square error: %.4f" % test_mse

    print "\n\n leasq select by zscore:"
    b2, z2, t2, p2,sig2_hat,se_beta_hat  = leasq_select_by_zscore(X, Y, X_t, Y_t, z_scores)
    print "Beta hat: %s" % b2
    print ""
    fs = "Z_scores:" + ", ".join(["%.4f"] * len(z2))
    print fs % tuple(z2)
    print ""
    print "train mean square error:", t2
    print "test mean square error: %.4f" % p2 
    
    f_v = f_value_check(test_mse, t2, len(beta), len(b2), len(X))
    print "\n\nf value check:", f_v
