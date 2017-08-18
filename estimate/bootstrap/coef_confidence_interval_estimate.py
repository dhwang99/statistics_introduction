#encoding: utf8

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot  as plt
import bisect
import pdb


'''
估计相关系数及相关系数估计的置信区间（和假设渐进正态的结果进行比较）
要画出直方图
'''

def coef(samples):
    dm = samples.T - samples.mean(axis=1)
    cov = np.dot(dm.T, dm)
    cf = cov[0, 1] / np.sqrt(cov[0,0]*cov[1,1])

    return cf
    

def estimate_coef(samples, B):
    x_means = np.zeros(B)
    N = samples.shape[1]
    for i in xrange(B):
        samples_B = np.zeros(samples.shape)
        for j in xrange(N):
            si = np.random.randint(0, N)
            samples_B[:, j] = samples[:, si]

        cf = coef(samples_B)
        x_means[i]  = cf
    
    return x_means
   
'''
这是一个错误的算法
'''
def get_interval_but_error(samples, alpha):
    samples.sort()
    m = samples.mean()
    cf = 1 - alpha

    mi = bisect.bisect_right(samples, m)
    
    N = samples.shape[0]
    li = mi
    hi = mi
    cover = 1
    while True:
        if li > 0:
            li -= 1
            cover += 1
        if hi < N-1:
            hi += 1
            cover += 1

        if cover*1./N >= cf:
            break
        
    return [samples[li], samples[hi]]

'''
find the min interval which cover (1-alpha) samples 
'''
def get_interval(samples, alpha):
    samples.sort()
    cf = 1 - alpha
    N = samples.shape[0] 
    sc = int(N*cf)
    min_ci = samples[N-1] - samples[0]
    for i in xrange(sc, N):
        ci = samples[i] - samples[i-sc]
        if ci < min_ci:
            min_ci = ci
            minid = i
        
    return [samples[minid - sc], samples[minid]]


if __name__ == "__main__":
    scores = np.array([[576,635,558,578,666,580,555,661,651,605,653,575,545,572], 
                   [3.39,3.30,2.81,3.03,3.44,3.07,3.00, 3.43,3.36,3.13,3.12,2.74,2.76,2.88]])

    B_list = np.array([5, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    
    coefs = None
    hat_mean = 0
    hat_se = 0

    for B in B_list:
        coefs = estimate_coef(scores, B)
        hat_mean, hat_se = coefs.mean(), coefs.std()

        print "Estimate mean and var, B=%s, mean=%.4f, se=%.4f" %(B, hat_mean, hat_se)

    #计算最后一次估计的置信区间(认为渐进正态)，画出直方图
   
    alpha = 0.05
    hat_se1 = hat_se * norm.ppf(1-alpha/2)

    print "confidence interval for B times is: [%.4f, %.4f]" % (max(0, hat_mean-hat_se1), min(1, hat_mean+hat_se1))

    group_count = int(np.sqrt(B))
    plt.hist(coefs, bins = group_count, normed=True, color='r', alpha=0.6)
    plt.savefig('images/coef_confidence.png', format='png')
    
    cf = get_interval(coefs, alpha)
    print "estimate confidence interval for B times is: [%.4f, %.4f]" % (cf[0], cf[1])
