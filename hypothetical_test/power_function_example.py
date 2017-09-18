#encoding: utf8

import numpy as np
from scipy.misc  import comb
from scipy.stats import norm
import matplotlib.pyplot as plt

import pdb

'''
test hypothetical: 用抽样的方法，来检查关于总体的H0假设(null hypothetical)能否被推翻，H1(备假设)能否被接受
'''

'''
HO: total <= c
P(X > c) = P(X in [c,n]) 
'''
def power_function_for_binomial(theta, n, c):
    beta_theta = 0.
    for i in xrange(c, n+1):
        beta_theta += comb(n, i) * np.power(theta, i) * np.power(1-theta, n - i)

    return beta_theta 


'''
在sigma已知的情况下，检验mu
单边检测
H0: mu <= mu0
H1: mu > mu0
alpha = 0.05 (or 0.01)

beta(mu) 
= Pmu(X_bar > c) 
= P(((X_bar-mu)*sqrt(n)/sigma) > (c-mu)*(sqrt(n)/sigma))
= P(((X_bar-mu)*sqrt(n)/sigma) > (c-mu)*(sqrt(n)/sigma))
=1 - Phi((c-mu)*sqrt(n)/sigma))

alpha = sup(beta(mu)) = 1 - Phi((c-mu0)*(sqrt(n)/sigma)) 
c = Phi_inv(1-alpha)*sigma/sqrt(n) + mu0

X_bar > c:
(X_bar - mu0)*sqrt(n)/sigma > Phi_inv(1-alpha)
Phi(X_bar - mu0)*sqrt(n)/sigma > 1-alpha
'''
def power_function_for_norm(mu, sigma, n, c):
    z = (c - mu)*np.sqrt(n)/sigma
    beta = 1 - norm.cdf(z)

    return beta

def c_norm(mu0, sigma, alpha, n):
    c = norm.ppf(1-alpha) * sigma/np.sqrt(n) + mu0
    return c

'''
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
    n = 5
    thetas = np.linspace(0, 1, 100)
    pw_vals = np.zeros(len(thetas))
    cs = [3, 5]
    
    for cid in xrange(len(cs)):
        c = cs[cid]
        for i in xrange(len(thetas)):
            pw_vals[i] = power_function_for_binomial(thetas[i], n, c)
        plt.plot(thetas, pw_vals, color=colors[cid])

    plt.savefig('images/binomial_power_function.png', format='png')
    plt.clf()

    n = 10
    alpha = 0.05
    sigma = 1.
    mus = np.linspace(-2, 3, 300)
    pw_vals = np.zeros(len(mus))
    mu0s =np.array([-1, 0, 1])

    for cid in xrange(len(mu0s)):
        mu0 = mu0s[cid]
        c = c_norm(mu0, sigma, alpha, n)
        for i in xrange(len(mus)):
            pw_vals[i] = power_function_for_norm(mus[i], sigma, n, c)
        plt.plot(mus, pw_vals, color=colors[cid])
        plt.plot([mu0, mu0], [0, alpha], color=colors[cid])

        print "norm hypothetical test, m0:%.4f; c:%.4f; alpha: %.4f" %(mu0, c, alpha)

    plt.plot([mus[0], mus[-1]], [alpha, alpha], color='r')

    plt.savefig('images/norm_power_function.png', format='png')

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
