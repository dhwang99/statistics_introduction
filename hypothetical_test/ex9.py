#encoding: utf8

import numpy as np
from scipy.stats import norm
import pdb

'''
wald p-value 
'''
X_bar = 195.27
se_X_bar = 5.0
Y_bar = 216.19
se_Y_bar = 2.4

delta_hat = Y_bar - X_bar
#delta = X-Y, se(delta) = sqrt(se(X)**2 + se(Y)**2) ~ sqrt(se_X_bar**2 + se_Y_bar**2)
se_delta_hat = np.sqrt(se_X_bar ** 2 + se_Y_bar ** 2)

W = np.abs(delta_hat - 0) / se_delta_hat

p = 2 * (1 - norm.cdf(W))
p = 2 * norm.cdf(-W)

print "W is: %.4f, p value is : %.4f"  % (W, p)


'''
char9. ex6
check people can postpone their death until after an important event

wald method
'''

from scipy.stats import binom

N = 1919
d = 922

p_hat = d*1./N
p0 = 0.5
delta = np.abs(p_hat - p0)

#se_delta = np.sqrt(p_hat*(1-p_hat)/N + p0*(1-p0)/N) , 这个是错的。 se是delta的，不需要加p0
se_delta = np.sqrt(p_hat*(1-p_hat)/N)
W = delta / se_delta

p_val = 2*norm.cdf(-W)

alpha = 0.05
inter = np.abs(se_delta * norm.ppf(alpha/2))
conf_interval = np.array([p_hat - inter, p_hat + inter])

print "ex6: by wald: p_val: %.4f" %(p_val)
print "ex6: interval: %.4f, %.4f" %(conf_interval[0], conf_interval[1])

'''
char9. ex7
proportion of three letter words:
 essays of Twain:
    .225 .262 .217 .240 .230 .229 .235 .217
 essays of Snodgrass:
    .209 .205 .196 .210 .202 .207 .224 .223 .220 .201
'''

twain_prop = np.array([.225,.262,.217,.240,.230,.229,.235,.217])
snod_prop = np.array([.209,.205,.196,.210,.202,.207,.224,.223,.220,.201])

#wald method
p_Xbar_hat = twain_prop.mean()
p_Ybar_hat = snod_prop.mean()

S_Xbar = twain_prop.std() / np.sqrt(len(twain_prop))
S_Ybar = snod_prop.std() / np.sqrt(len(snod_prop))

se_p_hat = np.sqrt(S_Xbar**2 + S_Ybar**2)
W_3lettle = np.abs((p_Xbar_hat - p_Ybar_hat)/se_p_hat)
p_val = 2 * norm.cdf(-W_3lettle)

print "Wald: p_value of owner of essays: %.4f" % (p_val)

'''
置换检验法
'''

tn = len(twain_prop)
sn = len(snod_prop)
T_abs = np.abs(p_Xbar_hat - p_Ybar_hat)

B = 1000
T_samples = np.zeros(B)
X_base = np.hstack((twain_prop, snod_prop))
for b in xrange(B):
    X_b = np.copy(X_base)
    np.random.shuffle(X_b)
    T_star = X_b[:tn].mean() - X_b[tn:].mean()
    T_samples[b] = T_star

#pdb.set_trace()
indicates = np.where(T_samples > T_abs)
I_count = len(indicates[0])
p_val = I_count * 1. / B
print "Permulation: p_value of owner of essays: %.4f" % (p_val)


'''
wald method. X ~ Possion(lambda)
'''

from scipy.stats import poisson

lamb0 = 1
n = 20
alpha = 0.05

B = 1000
rc = 0
z_a = norm.ppf(1-alpha/2)
for b in xrange(B): 
    samples = poisson.rvs(mu=lamb0, size=n)
    lamb_hat = samples.mean()
    W = np.abs((lamb_hat - lamb0) / np.sqrt(lamb_hat*1./n))
    if W > z_a:
        rc += 1

print "alpha: %.4f; total try: %s; reject: %s" % (alpha, B, rc)


