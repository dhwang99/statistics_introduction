#encoding: utf8

import numpy as np
from scipy.stats import norm

'''
X ~ N(mu, sigma), P(X < tau) = 0.95, get tau estimator
P((X-mu)/sigma < (tau - mu)/sigma) < 0.95
(tau - mu)/sigma = Z(0.95)
tau = z_095 * sigma + mu
'''

samples = np.array([3.23,-2.50,1.88,-0.68,4.43,0.17, \
    1.03,-0.07,-0.01,0.76,1.76,3.18, \
    0.33,-0.31,0.30,-0.61,1.52,5.43, \
    1.54,2.28,0.42,2.33,-1.03,4.00,0.39]); 

cf_val = 0.95
alpha = 0.05
n = len(samples)
z_cf = norm.ppf(cf_val)
z_alpha = norm.ppf(1-alpha/2)

#mle of mu, sigma
hat_mu = samples.mean()
hat_sigma = samples.std()

#估计的同变性
hat_tau = z_cf * hat_sigma + hat_mu 

#delta method. 手工算过，具体计算见笔记附件
hat_Jn = np.array([[hat_sigma**2, 0], [0, hat_sigma**2/2.]])/n
nambla_g = np.array([1, norm.ppf(cf_val)]) 
hat_se_hat_tau = np.dot(np.dot(nambla_g, hat_Jn), nambla_g)  
hat_se_hat_tau = np.sqrt(hat_se_hat_tau)
confidence_interval = np.array([hat_tau - hat_se_hat_tau * z_alpha, \
                               hat_tau + hat_se_hat_tau * z_alpha])

print "Delta: hat_tau:%.4f, hat_se_hat_tau:%.4f, confidence_interval: %.4f, %.4f" % \
       (hat_tau, hat_se_hat_tau, confidence_interval[0], confidence_interval[1])

#parameter bootstrap
B = 1000
B_tau_samples = np.zeros(B)

for b in xrange(B):
    x_b_samples = norm.rvs(loc=hat_mu, scale=hat_sigma, size=n)
    b_sample = z_cf * x_b_samples.std() + x_b_samples.mean() 
    B_tau_samples[b] = b_sample

hat_tau = B_tau_samples.mean()
hat_se_hat_tau = B_tau_samples.std() 
confidence_interval = np.array([hat_tau - hat_se_hat_tau * z_alpha, \
                               hat_tau + hat_se_hat_tau * z_alpha])

print "Bootstrap: hat_tau:%.4f, hat_se_hat_tau:%.4f, confidence_interval: %.4f, %.4f" % \
       (hat_tau, hat_se_hat_tau, confidence_interval[0], confidence_interval[1])
