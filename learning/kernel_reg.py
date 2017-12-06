#encoding: utf8

import numpy as np
import pdb
from scipy.stats import f as f_stats
from scipy.stats import norm 
import matplotlib.pyplot as plt

from data_loader import load_data, normalize_data
from cv_common import gen_CV_samples_by_K_folds

'''
X_train: 生成 h/bandwidth

X_test: 评价结果(或者用交叉验证的方法来选)


从试验结果看， 新核和旧核差别不明显
此外，还有一些别的核函数可以尝试一下。另一个，看起来核函数实现得也有些问题？
by wdh:看起来是的，针对每一维应该有不同的h(只是我们用的是归一化后的数据，如果用h/si, 用归一化的数据，h用一个就可以了。不过实际上应该还是多个hi, 这样各维度的权重就不一样了？)
'''
'''
old kernels
kernels = {
        #x_s: x_star, 
     'box': lambda x:1./(2**len(x)) if (len(np.where(np.abs(x)>1.)[0]) == 0) else 0.,    #f(x) = c,
    'rbf':lambda x:np.exp(-sum(x**2)/2),   #'norm':lambda x:np.prod(norm.pdf(x)),
    'ep':lambda x:np.prod(3./4*(1-x**2)) if (np.where(np.abs(x)>1)[0].shape[0] == 0) else 0,
    'cos':lambda x: np.prod(1./2*np.cos(x)) if (np.where(np.abs(x)>np.pi/2)[0].shape[0] == 0) else 0.,
}
'''

kernels = {
     'box': lambda x:1./2 if (np.abs((x).sum()) <= 1) else 0.,    #f(x) = c,
    'rbf':lambda x:np.exp(-np.linalg.norm(x,2)/2),   #'norm':lambda x:np.prod(norm.pdf(x)),
    'ep':lambda x:3./4*(1-(x**2).sum()) if ((x**2).sum()<=1) else 0,
    'cos':lambda x: 1./2*np.cos(np.sum(x)) if (np.abs((x).sum()) <= 1) else 0.,
}

def fit(X_train, Y_train, x, h, kf):
    base_each = map(lambda i:1.0/h*kf((X_train[i]-x)/h), range(len(Y_train))) 
    weight_each = map(lambda i:1.0/h*kf((X_train[i]-x)/h)*Y_train[i], range(len(Y_train)))

    weight = np.sum(np.array(weight_each))
    base = np.sum(np.array(base_each))
    
    if base >= 1e-5:
        return weight/base
    else:
        return 0.


def kernel_reg(kf):
    train_X, train_Y, test_X, test_Y, dt_conv_fun = load_data(type=0, need_bias=0, y_standard=0)
    N = 50
    mse_mean = np.zeros(N)
    mse_std = np.zeros(N)

    K = 10
    K_folds = gen_CV_samples_by_K_folds(train_X, train_Y, K)
    steps = np.logspace(-3, 3, N)

    for ni in xrange(N):
        h = steps[ni] 
        mse_k = np.zeros(K)

        for k in xrange(K):
            ks = K_folds[k]
            X_CV, Y_CV, Xt_CV, Yt_CV,nf = normalize_data(ks[0], ks[1], ks[2], ks[3])
            
            rx = np.zeros(len(Yt_CV))
            for i in xrange(len(Yt_CV)):
                rx[i] = fit(X_CV, Y_CV, Xt_CV[i], h, kf)

            mse_k[k] = np.sum((rx - Yt_CV) ** 2) / len(Y_CV)
        
        mse_mean[ni] = mse_k.mean()
        mse_std[ni] = mse_k.std()/np.sqrt(K)
    
    best_id = np.argmin(mse_mean)
    best_h = steps[best_id]

    X, Y, X_t, Y_t, dt_conv_fun = load_data()
    test_rst = np.zeros(len(Y_t))

    for i in xrange(len(Y_t)):
        test_rst[i] = fit(X, Y, X_t[i], best_h, kf)

    test_mse = np.sum((test_rst - Y_t)**2)/len(Y_t)
    test_std = test_rst.std()

    train_rst = zip(steps, mse_mean, mse_std)

    return train_rst, (best_h, test_mse, test_std)


if __name__ == '__main__':
    for kname, kernel in kernels.items():
        print "\nProcess %s:" % kname
        train_rst, test_rst = kernel_reg(kernel)
        for r in train_rst:
            print "h: %.4f  mse_mean: %.4f  mse_std: %.4f" % r
        print "Test rst: h: %.4f mse: %.4f  std: %.4f" % test_rst

