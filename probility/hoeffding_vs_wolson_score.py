#encoding: utf8

import numpy as np
from scipy.stats import norm

'''
P(Xi \in [ai,bi]) = 1
X_bar = sum(Xi)/n
P((X_bar - E(X_bar) >= t) <= exp(-2*t**2 * n**2)/(sum(bi - ai)**2)
P(|(X_bar - E(X_bar)| >= t) <= 2 * exp(-2*t**2 * n**2)/(sum(bi - ai)**2)
'''

def hoeffding_inequal(t, n, a, b):
    val = 2 * (t**2) * (n**2)
    s = ((b-a)**2) * n

    return np.exp(-val/s)

def wilson_score_interval(n, s, alpha):
    p_hat = s*1./n
    z = norm.ppf(1-alpha/2)
    part = z * np.sqrt(z**2 - 1./n + 4 * n * p_hat * (1-p_hat) + (4 * p_hat-2)) + 1
    w_down = np.max([0, (2 * s + z ** 2 - part)/(2*(n+z**2))])
    w_up = np.min([1, (2 * s + z ** 2 + part)/(2*(n+z**2))])

    return [w_down, w_up]

def comp_wilson_vs_normal(n, p, alpha):
    print ""
    a_list = [0.05, 0.10, 0.20, 0.30, 0.40]
    
    hat_sigma = np.sqrt(p * (1-p)/n)
    for alpha in a_list:
        '''
        confidence interval by sim normal
        '''
        dis = norm.ppf(1-alpha) * hat_sigma
        print "approximately to normal: p: %.4f, alpha: %.4f, interval: %.4f, %.4f" % (p, alpha, p-dis, p+dis)

        '''
        wilson score
        '''
        interval = wilson_score_interval(n, n*p, alpha)
        print "wilson score: alpha: %.4f, interval: %.4f, %.4f" % (alpha, interval[0], interval[1])


'''
binomal test
'''

p = 0.16
n = 50
a = 0
b = 1

t_list = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
t2_list = -t_list


for t in t_list:
    hoef = hoeffding_inequal(t, n, a, b)
    print "hoef t: %.4f, hoef: %.4f" % (t, hoef)

for t in t2_list:
    hoef = hoeffding_inequal(t, n, a, b)
    print "hoef t: %.4f, hoef: %.4f" % (t, hoef)

a_list = [0.05, 0.10, 0.20, 0.30, 0.40]
comp_wilson_vs_normal(n, p, a_list)

p = 0.5
comp_wilson_vs_normal(n, p, a_list)

p = 0.6
comp_wilson_vs_normal(n, p, a_list)

p = 0.6
n = 60
comp_wilson_vs_normal(n, p, a_list)
