#encoding: utf8

import numpy as np
import pdb
from scipy.stats import f as f_stats
from scipy.stats import norm 
import matplotlib.pyplot as plt

from data_loader import load_data
from cv_common import gen_CV_samples_by_K_folds

'''
X_train: 生成 h/bandwidth

X_test: 评价结果(或者用交叉验证的方法来选)
'''

kernels = {
    'box': lambda x:1./(2**len(x)) if (len(np.where(np.abs(x)>1.)) == 0) else 0.,    #f(x) = c,
    'norm':lambda x:np.prod(norm.pdf(x)),
    'ep':lambda x:np.prod(3./4*(1-x**2)) if (np.where(np.abs(x)>1)[0].shape[0] == 0) else 0,
    'cos':lambda x: np.prod(1./2*np.cos(x)) if (np.where(np.abs(x)>np.pi/2)[0].shape[0] == 0) else 0.,
}

def kernel_reg(X_train, Y_train, X_test, Y_test, kf):
    N = 50
    mse_mean = np.zeros(N)
    mse_std = np.zeros(N)

    K = 10
    K_folds = gen_CV_samples_by_K_folds(X_train, Y_train, K)
    steps = np.logspace(-3, 3, N)

    for ni in xrange(N):
        h = steps[ni] 
        mse_k = np.zeros(K)

        for k in xrange(K):
            ks = K_folds[k]
            X_CV, Y_CV, Xt_CV, Yt_CV = ks
            
            rx = np.zeros(len(Yt_CV))
            for i in xrange(len(Yt_CV)):
                xi = Xt_CV[i]
                yi = Yt_CV[i]

                yp = map(lambda i:kf((X_CV[i]-xi)/h)*Y_CV[i], xrange(len(Y_CV)))
                base = map(lambda i:kf((X_CV[i]-xi)/h), xrange(len(Y_CV)))

                rx[i] = np.array(yp).sum() / np.array(base).sum()

            mse_k[k] = np.sum((rx - Yt_CV) ** 2) / len(Y_CV)

        mse_mean[ni] = mse_k.mean()
        mse_std[ni] = mse_k.std()/np.sqrt(K)

    return zip(steps, mse_mean, mse_std)


if __name__ == '__main__':
    X, Y, X_t, Y_t,dt = load_data()
    
    for kname, kernel in kernels.items():
        print "\nProcess %s:" % kname
        rst = kernel_reg(X, Y, X_t, Y_t, kernel)
        for r in rst:
            print "h: %.4f  mse_mean: %.4f  mse_std: %.4f" % r

