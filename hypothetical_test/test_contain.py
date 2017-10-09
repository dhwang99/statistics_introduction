#encoding: utf8

import numpy as np
from scipy.misc  import comb
from scipy.stats import norm
import matplotlib.pyplot as plt

import pdb

'''
容量和alpha, beta都有关, 和delta有关。一般delta取一个sigma
err1 = alpha = 0.1
err2 = beta = 0.2

正态样本容量，用来控制第二类错误(这么说还好不对)
delta 默认为一个标准差

Phi((c - mu0)*sqrt(n)/sigma) <= (1-alpha)
c <= ppf(1-alpha) * sigma/sqrt(n) + mu0

ds = delta / sigma  #delta,sigma指定后，这个就确定了。引入这个变量主要是方例后继的计算
mu = mu0 + delta
Phi((c - mu)*sqrt(n)/sigma) < beta
Phi((c - mu)*sqrt(n)/sigma) <  beta
Phi((c - mu0 - delta)*sqrt(n)/sigma) < beta
(c - mu0 - delta)*sqrt(n)/sigma) < Phi(beta)
(c - mu0 - delta)*sqrt(n)/sigma) < -Phi(1-beta)

#c取最大值，有
(Za * sigma/sqrt(n) + mu0 - mu0 - delta)*sqrt(n)/sigma) < -Zbeta
Za - delta*sqrt(n)/sigma < -Zbeta
sqrt(n) > (Za + Zbeta) * sigma/delta
'''
def norm_sample_contain(sigma, delta, max_alpha=0.1, max_beta=0.2):
    if delta == None:
        delta = sigma
   
    Za =  norm.ppf(1-max_alpha)
    Zb = norm.ppf(1-max_beta)

    min_sqrt_n = (Za + Zb) * sigma / delta 
    n = np.ceil(min_sqrt_n ** 2)

    return n

'''
计算正态分布下的p值
'''
def p_value():

    return None

if __name__ == "__main__":
    colors = ['g', 'b', 'k']
    #test contain of samples
    mu0 = 0
    sigma = 1.
    
    betas = np.linspace(0.01, 0.3, num=50)
    contains = np.zeros(len(betas))
    for i in xrange(len(betas)): 
        beta = betas[i]
        n = norm_sample_contain(sigma, delta=sigma, max_alpha=0.1, max_beta=beta)
        contains[i] = n

    plt.clf()
    plt.plot(betas, contains, color='r')
    print "betas:", betas
    print "n:", contains

    for i in xrange(len(betas)): 
        beta = betas[i]
        n = norm_sample_contain(sigma, delta=sigma, max_alpha=0.05, max_beta=beta)
        contains[i] = n
    plt.plot(betas, contains, color='k')
    print "betas:", betas
    print "n:", contains

    plt.savefig('images/norm_contain.png', format='png')
