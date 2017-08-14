#encoding: utf8

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import cauchy 

import pdb

'''
生成norm, cauchy的样本(loc=0, scale=1), 取alpha=0.05, 检验edf对真值的覆盖情况
共验证了一千次，比例略高于95%. 这个主要是因为 hoeffding 不等式条件相对较松导致

因为分布来自正态/cauche, 改用gauss算了下. 分布更宽(不用能正态分布来计算置信区间，估计无法说明是渐进正态的)
'''

def check_cdf_estimate(stat, M, N):
    alpha = 0.05
    epsilon = np.sqrt(1./(2*N)*np.log(2/alpha))  #P(Psupx(|hat_F - F| > epsilon) <= 2*exp(-2n*epsilon**2) = alpha
    not_hit = 0
    #epsilon = 2 * 1/np.sqrt(N*1.)
    
    for i in xrange(M):
        samples = stat.rvs(size=N, loc=0, scale=1)
        sorted_data = np.sort(samples)
        cdfs = stat.cdf(sorted_data) 

        hat_px = 0
        for i in xrange(N):
            hat_px += 1./N 
            if np.abs(cdfs[i] - hat_px) >= epsilon:
                not_hit += 1
                break

    fit_ratio = 1 - not_hit * 1./M
    print "estimate cdf for %s, alpha: %.3f, samples: %s, epsilon: %.3f, real fit: %.3f." %  \
        (stat.name, alpha, N, epsilon, fit_ratio)

    return fit_ratio


def cal_and_plot_confidence_band(samples, cdf, color='g', only_samples=False, alpha=0.05):
    N = len(samples)
    epsilon = np.sqrt(1./(2*N)*np.log(2/alpha))  #P(Psupx(|hat_F - F| > epsilon) <= 2*exp(-2n*epsilon**2) = alpha
    
    sorted_s = np.sort(samples)
    
    tc = 0
    x_points = np.zeros(N)
    px = np.zeros(N)

    for i in xrange(N):
        x = sorted_s[i]
        tc += 1.
        px[i] = tc/N
        x_points[i] = x
    
    x2_points = np.zeros(N * 2+2)
    px2 = np.zeros(N*2+2)

    x2_points[0] = x_points[0] - 0.5
    px2[0] = 0.
    for i in xrange(0, N):
        x2_points[2*i+1] = x_points[i]
        px2[2*i+1] = px2[2*i]

        x2_points[2*i+2] = x_points[i]
        px2[2*i+2] = px[i]

    x2_points[N*2+1]  = x2_points[N*2] + 0.5
    px2[2*N+1] = 1

    up_band = px2 + epsilon
    down_band = px2 - epsilon

    up_band[up_band>1] = 1
    down_band[down_band<0] = 0
    
    plt.plot(x2_points, px2, color=color)
    if only_samples == False:
        plt.plot(x_points, cdf, color='r')
        plt.plot(x2_points, up_band, color='b')
        plt.plot(x2_points, down_band, color='k')

if __name__ == "__main__":
    N=100
    for i in xrange(1000):
        samples = norm.rvs(size=N, loc=0, scale=1)
        cal_and_plot_confidence_band(samples, None, color='c', only_samples=True)

    samples = norm.rvs(size=N, loc=0, scale=1)
    samples = np.sort(samples)
    cdf = norm.cdf(samples, loc=0, scale=1)
    cal_and_plot_confidence_band(samples, cdf)

    plt.savefig('images/confidence_band_norm_%s.png'%N, format='png')

    check_cdf_estimate(norm, 1000, 100)
    check_cdf_estimate(cauchy, 1000, 100)

    print ""

    check_cdf_estimate(norm, 1000, 1000)
    check_cdf_estimate(cauchy, 1000, 1000)
