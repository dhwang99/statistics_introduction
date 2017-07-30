#encoding: utf8

from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

import pdb

'''
I = \int h(x)f(x)dx = \int h(x)f(x)g(x)/g(x) dx = \int (h(x)f(x)/g(x) g(x) dx
X ~ g(X)
\hat I = 1/N * \sum {h(Xi)f(Xi)/g(Xi)}
'''

'''
P(Z>3), Z ~ N(0, 1) 
P(Z>3) = 0.0013

从例子也可以看出来，当 g(x)与f(x)区别比较大时，importance_sampling误差就比较大了

用有偏估计效果似乎好一些：实际更差。

\hat I = \sum {h(Xi)f(Xi)/g(Xi)} / \sum {f(Xi)/g(Xi)}, 即不用N, 而用权重和表示：类似EM算法

'''

def norm_pdf(x, mu, sigma):
    return norm.pdf((x - mu)/sigma)

def get_hat_I(g_mu, g_sigma, N):
    f_mu = 0.
    f_sigma = 1.

    g_samples = np.random.normal(g_mu, g_sigma, N)
    
    h_1_samples = []
    samples_sum = 0.
    weight_sum = 0.
    for x in g_samples:
        if x > 3:
            h_1_samples.append(x)   
            weight = norm_pdf(x, f_mu, f_sigma) / norm_pdf(x, g_mu, g_sigma) 
            samples_sum += 1. * weight 
            weight_sum += weight
    hat_I = samples_sum/N
    if weight_sum == 0:
        weight_sum = 1.
    hat_I_weight = samples_sum / weight_sum

    return hat_I, hat_I_weight 

def test_truncated_norm(g_mu, g_sigma, N, test_count):
    hat_I_lst = []
    hat_I_weight_lst = []
    for i in xrange(test_count):
        hat_I, hat_I_w = get_hat_I(g_mu, g_sigma, N)
        hat_I_lst.append(hat_I)
        hat_I_weight_lst.append(hat_I_w)

    return np.array(hat_I_lst), np.array(hat_I_weight_lst)

if __name__ == "__main__":
    N = 100
    test_count = 100
    real_val = 0.0013

    g_mu = 0.
    g_sigma = 1.
    params = [(0., 1.), (4., 1.), (4., 2.), (4., 4)] 
    for g_mu, g_sigma in params:
        hat_vals, hat_vals_w = test_truncated_norm(g_mu, g_sigma, N, test_count) 
        print "\nreal value: %s; g_mu:%s; g_sigma:%s; hat_mean: %s; hat_std: %s; weighted_mean:%s; weighted_std: %s " %  \
             (real_val, g_mu, g_sigma, hat_vals.mean(), hat_vals.std(), hat_vals_w.mean(), hat_vals_w.std())

