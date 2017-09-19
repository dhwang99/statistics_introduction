#encoding: utf8

import numpy as np
from scipy.stats import t as t_stat
from scipy.stats import norm

import pdb

'''
H0: mu_hat = mu0; H1: mu_hat != mu0

X ~ N(mu, sigma), mu,sigma都未知

wald test: 适合大样本;
t test: 适合小样本

wald test:
    W = sqrt(n)*(mu_hat - mu0)/se_hat, se_hat: 公式; 样本方差代替;  bootstrap方法
    |W| > phi(alpha/2), reject

t test:
    T = sqrt(n)*(mu_hat - mu0)/se_hat, se_hat(样本方差代替)
    |T| > t(n-1, alpha/2), reject

    t-test 自由度需要注意一下

    在n=10时，有时候会把真值拒绝

'''

def wald_test(samples, mu0, alpha):
    mu_hat = samples.mean()
    se_hat = samples.std()

    W = (mu_hat - mu0) * np.sqrt(len(samples)) / se_hat
    z_a = norm.ppf(1-alpha/2.)

    print "W:%.4f; alpha: %.4f; z_a: %.4f" % (W, alpha, z_a)

    if np.abs(W) > norm.ppf(1-alpha/2.):
        return True

    return False

def t_test(samples, mu0, alpha):
    mu_hat = samples.mean()
    se_hat = samples.std()

    T = (mu_hat - mu0) * np.sqrt(len(samples)) / se_hat
    t_n = t_stat.ppf(1-alpha/2., len(samples) - 1)

    print "T:%.4f; alpha: %.4f; t_n: %.4f" % (T, alpha, t_n)
    
    if np.abs(T) > t_n:
        return True

    return False

alpha = 0.05
samples1 = norm.rvs(size=10)
samples2 = norm.rvs(size=50)
mu0s = np.linspace(-5, 5, 11)
for mu0 in mu0s:
    print "mu0:", mu0
    rst = t_test(samples1, mu0, alpha)
    print "t_test result,", rst

    rst = wald_test(samples2, mu0, alpha)
    print "wald_test result,", rst

    print ""

