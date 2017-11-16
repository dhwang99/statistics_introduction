#encoding: utf8

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pdb

'''
用一维数据模拟
用交叉验证进行选择h
从结果看，核的选择差别不大, h影响大
'''
kernels = {
    'box': lambda x:1./2 if (np.abs(x)<=1) else 0.,
    'norm':lambda x:norm.pdf(x),
    'ep':lambda x:3./4*(1-x**2) if (np.abs(x)<=1) else 0,
    'cos':lambda x: 1./2*np.cos(x) if (np.abs(x)<=np.pi/2) else 0.,
}


def fit(trainX, trainY, testX, h=1., kn='norm'):
    kfun = kernels[kn]
    rx = np.zeros(len(testX))
    for i in xrange(len(testX)):
        x = testX[i]
        base = map(lambda j:kfun((trainX[j]-x)/h), xrange(len(trainY)))
        p = map(lambda j:kfun((trainX[j]-x)/h)*trainY[j], xrange(len(trainY)))

        rx[i] = np.array(p).sum() / np.array(base).sum()

    return rx

def test_kernel(X, Y, X_t, Y_t, kn, outdetail=False):
    err_best=1e10
    rx_best = np.zeros(len(X_t))
    h_best = 0

    for h in np.logspace(-3, 3, 50):
        rx = fit(X, Y, X_t, h, kn)
        err = Y_t - rx
        rss = np.dot(err, err)
        if err_best > rss:
            err_best = rss 
            rx_best = np.copy(rx)
            h_best = h
        
        if outdetail:
            print "h: %.4f rss: %.4f" % (h, rss)

    print "best fit: h: %.4f err_best: %.4f" % (h_best, err_best) 
    plt.clf()
    plt.scatter(trainX, trainY,color='black')
    plt.scatter(testX, rx_best,color='blue',linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.savefig('images/kr_1d_%s.png' % kn)

    return h_best, err_best


n = 200
tn = int(n * 0.75)
X = np.linspace(0, 9.9, n)
np.random.shuffle(X)
noise = np.random.normal(0, 0.5, n)
Y = X * 2 + noise + 3
Y = np.cos(X) * 10 + noise 

trainX = X[:tn]
trainY = Y[:tn]

testX = X[tn:]
testY = Y[tn:]

check_rst = {}
for kn in kernels:
    print ""
    print "test kernel: %s" % kn
    rst = test_kernel(trainX, trainY, testX, testY, kn, outdetail=True)
    
    check_rst[kn] = rst

print ""
print "CHECK RST:"
for kn,rst in check_rst.items():
    print "kernel: %s best fit: h: %.4f err_best: %.4f" % (kn, rst[0], rst[1]) 
