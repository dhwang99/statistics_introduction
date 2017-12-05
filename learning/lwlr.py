# encoding: utf8


'''
locally weighted linear regression

J = sum_i(W_i * (y_i - seta * x_i)**2)

W_i = exp(-np.linalg.norm(x_i - x)/(2*k**2))

seta = (X.T * W * X).inv * X.T * W * y

'''

import numpy as np
import matplotlib.pyplot as plt
import pdb

from cv_common import gen_CV_samples_by_K_folds 
from data_loader import load_data

def lwlr(x_i, train_X, train_Y, k):
    m,n = train_X.shape
    
    W = np.zeros((m,m))
    for i in xrange(m):
        W[i,i] = np.exp(-1. * np.linalg.norm(x_i - train_X[i], 2)/(2*k))
    
    X = train_X
    Y = train_Y
    XTW = np.dot(X.T, W)
    XTWX = np.dot(XTW, X)
    XTWy = np.dot(XTW, Y)

    seta_hat = np.dot(np.linalg.inv(XTWX), XTWy)

    return np.dot(seta_hat, x_i)

def lwlrs(xs, train_X, train_Y, k):
    m,n = xs.shape
    y_hats = np.zeros(m)

    for i in xrange(m):
        xi = xs[i]
        y_hats[i] = lwlr(xi, train_X, train_Y, k)

    return y_hats


'''
基于X,Y数据，用k-fold方法选取出最优的k
'''
def select_k(X, Y, log_beg=0, log_end=3, imgpath=None):
    CV_K = 10
    k_size = 100
    cv_samples = gen_CV_samples_by_K_folds(X, Y, CV_K)

    ks = np.logspace(log_beg, log_end, k_size)
    K_mses = np.zeros(k_size)
    K_stds = np.zeros(k_size)

    for ki in xrange(k_size):
        k = ks[ki]
        cv_mses = np.zeros(CV_K) 

        for ci in xrange(CV_K):
            X_, Y_, X_test, Y_test = cv_samples[ci]
            y_hats = lwlrs(X_test, X_, Y_, k)
            y_err = y_hats - Y_test

            cv_mses[ci] = np.dot(y_err, y_err) / len(y_err)

        K_mses[ki] = cv_mses.mean()
        K_stds[ki] = cv_mses.std()/np.sqrt(CV_K) 

        print "CV for K: %.4f, mse: %.4f, std: %.4f" % (k, K_mses[ki], K_stds[ki])

    best_kid = np.argmin(K_mses)
    
    if imgpath != None:
        plt.errorbar(ks, K_mses, yerr=K_stds, fmt='-o')
        plt.savefig(imgpath)

    return ks[best_kid]

def test_1d():
    data = np.loadtxt('data/ex0.txt')
    np.random.shuffle(data)

    n,m = data.shape

    tlen = int(n * 0.7)
    train_data = data[:tlen]
    test_data = data[tlen:]

    train_X = train_data[:, :-1]
    train_Y = train_data[:,-1]

    test_X = test_data[:,:-1]
    test_Y = test_data[:,-1]

    best_k = select_k(train_X, train_Y, log_beg=-3., log_end=3., imgpath='images/lwlr_1d_errorbar.png')
    y_hats = lwlrs(test_X, train_X, train_Y, best_k)

    p_errs = y_hats - test_Y
    test_mse = np.dot(p_errs, p_errs)/len(test_Y)
    test_std = p_errs.std()
    #pdb.set_trace()

    print ""
    print "Predict, k: %.4f, mse: %.4f, std: %.4f" % (best_k, test_mse, test_std)

   
def test_course():
    train_X, train_Y, test_X, test_Y, dt_stand_fun = load_data(type=0, need_bias=1, y_standard=0)

    best_k = select_k(train_X, train_Y, log_beg=0, log_end=4, imgpath='images/lwlr_prostate_errorbar.png')
    y_hats = lwlrs(test_X, train_X, train_Y, best_k)

    p_errs = y_hats - test_Y
    test_mse = np.dot(p_errs, p_errs)/len(test_Y)
    test_std = p_errs.std()

    print ""
    print "Predict, k: %.4f, mse: %.4f, std: %.4f" % (best_k, test_mse, test_std)

if __name__ == "__main__":
    print "test 1d data:"
    test_1d()

    print "\n test course data:"
    test_course()

