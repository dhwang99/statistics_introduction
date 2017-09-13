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
total < H0: accept H0

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

    plt.plot([mus[0], mus[-1]], [alpha, alpha], color='r')

    plt.savefig('images/norm_power_function.png', format='png')
