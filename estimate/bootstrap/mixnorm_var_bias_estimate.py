#encoding: utf8

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

'''
已知样本来自于以下混合高斯分布

F = 0.2 * N(1, 2**2) + 0.8 * N(6, 1)

估计方差、偏差. 这样就估计出mse了

X ~ F
X_bar ~ 4.997
'''

'''
这种产生mix_norm的方法靠谱不？
'''
def gen_mix_norm_samples(N):
    mus = np.array([1, 6])
    sigmas = np.array([2, 1])
    mix = np.array([0.2, 0.8])
    mix_num = mus.shape[0]
    each_samples = np.zeros((mix_num, N))
    for i in xrange(mix_num):
        each_samples[i] = norm.rvs(size=N, loc=mus[i], scale=sigmas[i])

    samples = np.zeros(N)
    for i in xrange(N):
        samples[i] = np.dot(mix, each_samples[:, i]) 

    return samples

def estimate_var_bias_by_bootstrap(samples, B):
    N = samples.shape[0]
    X_mean = samples.mean()
    X_mean_samples = np.zeros(B)
    for bid in xrange(B):
        b_samples = np.zeros(N)
        for nid in xrange(N):
            s = np.random.randint(0, N)
            b_samples[nid] = samples[s]

        X_mean_samples[bid] = b_samples.mean()

    boot_X_mean = X_mean_samples.mean()
    boot_X_var = X_mean_samples.var()
    bias = boot_X_mean - X_mean

    print "bootstrap for B:%s, mean: %4f, boot_mean: %.4f, std: %.4f, bias: %.4f" %\
            (B, X_mean, boot_X_mean, np.sqrt(boot_X_var), bias)

    return


def estimate_var_bias_by_knife(samples, miss_num=1):
    N = samples.shape[0]
    X_mean = samples.mean()
    X_mean_samples = np.zeros(N)

    for bid in xrange(N):
        b_samples = np.hstack((samples[:bid], samples[bid+1:]))
        X_mean_samples[bid] = b_samples.mean()

    kinfe_X_mean = X_mean_samples.mean()
    error_var = X_mean_samples.var()
    kinfe_X_var = (N-1)*1./N * np.dot(X_mean_samples - X_mean, X_mean_samples - X_mean)
    bias = kinfe_X_mean - X_mean

    '''
    加一个错误的输出。如果直接用 X_mean_samples.var()求样本标准差，和真实的标准差差别太大. 所以一定要用标准的方法计算
    '''
    print "kinfe mean: %4f, knife_mean: %.4f, std: %.4f, std(error): %.4f, bias: %.4f" %\
            (X_mean, kinfe_X_mean, np.sqrt(kinfe_X_var), np.sqrt(error_var), bias)

    return


if __name__ == "__main__":
    B_list = np.array([5, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    samples = gen_mix_norm_samples(200)
    for B in B_list:
        estimate_var_bias_by_bootstrap(samples, B)
    
    print ""
    estimate_var_bias_by_knife(samples, B)
