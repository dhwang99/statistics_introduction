#encoding: utf8

import numpy as np
from scipy.stats import norm
import pdb

def estimate_theta_by_bootstrap(mu, sigma, N, B):
    T_stars = np.zeros(B)
    params = norm.rvs(size=N, loc=mu, scale=sigma)

    for b in xrange(B):
        sids = np.random.randint(low=0, high=N, size=N)
        X_stars = np.take(params, sids)
        T_stars[b] = np.exp(X_stars.mean())

    return T_stars 
        

if __name__ == '__main__':
    B = 1000
    N = 100
    mu = 5
    sigma = 1

    T_stars = estimate_theta_by_bootstrap(mu, sigma, N, B)

    T_stars.sort()
    #estimate error estimate:
    T_mean = T_stars.mean() 
    T_se = np.sqrt(T_stars.var())
    print "se is: %.5f" % T_se
    
    alpha = 0.05
    #estimate confidence interval:
    #Normal:
    C_a = T_mean - norm.ppf(1-alpha/2) * T_se
    C_b = T_mean - norm.ppf(alpha/2) * T_se
    print "Normal confidence interval: [%.5f, %.5f]" % (C_a, C_b)

    #percentile:
    p_a = np.percentile(T_stars, alpha/2 * 100)
    p_b = np.percentile(T_stars, (1-alpha/2) * 100)
    print "percentile confidence interval: [%.5f, %.5f]" % (p_a, p_b)

    #pivotal:
    pi_a = T_mean*2 - p_b
    pi_b = T_mean*2 - p_a
    print "pivotal confidence interval: [%.5f, %.5f]" % (pi_a, pi_b)



