#encoding: utf8

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm 
import pdb

'''
X1, X2, ..., Xn ~ N(mu, 1)
seta = exp(mu)

get hat_se(seta), 0.9 confience interval of seta, using delta
get hat_se(seta) by delta, parameter bootstrap, no parameter bootstrap 

compare delta, param boostrap, nonbootstrap, we find delta and param boostrap is more stable
'''

mu = 5
sigma = 1
n = 100
B = 10000
group_count = int(np.sqrt(B))

samples = norm.rvs(loc=mu, scale=sigma, size=n)
ppf_0_95 = norm.ppf(0.975, loc=mu, scale=sigma)

#mle and delta
X_mean = samples.mean()
hat_seta = np.exp(X_mean)
hat_se_seta_delta = np.sqrt((sigma**2) * 1./n) * np.exp(X_mean) 
cn = [hat_seta - ppf_0_95 * hat_se_seta_delta, hat_seta + ppf_0_95 + hat_se_seta_delta]

print "delta method: hat_seta: %.4f; hat_se: %.4f; confience interval: [%.4f, %.4f]" % \
        (hat_seta, hat_se_seta_delta, cn[0], cn[1])

delta_samples = norm.rvs(loc=hat_seta, scale=hat_seta, size=B)
plt.clf()
plt.hist(delta_samples, bins = group_count, color='b', alpha=0.6)
plt.savefig('images/char9_9_delta.png', format='png')


#parameter bootstrap
param_B_samples = np.zeros(B)
for b in xrange(B):
    # X ~ (log(hat_seta), sigma)
    b_samples = norm.rvs(loc=np.log(hat_seta), scale=sigma, size=n) 
    param_B_samples[b] = np.exp(b_samples.mean())

hat_seta = param_B_samples.mean()
hat_se_seta = param_B_samples.std()
cn = [hat_seta - ppf_0_95 * hat_se_seta, hat_seta + ppf_0_95 + hat_se_seta]

print "parameter bootstrap method: hat_seta: %.4f; hat_se: %.4f; confience interval: [%.4f, %.4f]" % \
        (hat_seta, hat_se_seta, cn[0], cn[1])

plt.clf()
plt.hist(param_B_samples, bins = group_count, color='b', alpha=0.6)
plt.savefig('images/char9_9_param.png', format='png')

#nonparameter bootstrap
noparam_B_samples = np.zeros(B)
for b in xrange(B):
    bids = np.random.randint(0, high=n, size=n)
    noparam_B_samples[b] = np.exp(np.take(samples, bids).mean())

hat_seta = noparam_B_samples.mean()
hat_se_seta = noparam_B_samples.std()
cn = [hat_seta - ppf_0_95 * hat_se_seta, hat_seta + ppf_0_95 + hat_se_seta]

print "nonparameter bootstrap method: hat_seta: %.4f; hat_se: %.4f; confience interval: [%.4f, %.4f]" % \
        (hat_seta, hat_se_seta, cn[0], cn[1])

plt.clf()
plt.hist(noparam_B_samples, bins = group_count, color='b', alpha=0.6)
plt.savefig('images/char9_9_noparam.png', format='png')

