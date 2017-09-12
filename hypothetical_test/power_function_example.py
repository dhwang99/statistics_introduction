#encoding: utf8

import numpy as np
from scipy.misc  import comb
from scipy.stats import norm
import matplotlib.pyplot as plt

'''
test hypothetical: 用抽样的方法，来检查关于总体的H0假设(null hypothetical)能否被推翻，H1(备假设)能否被接受

'''

'''
total < H0: accept H0

'''
def power_function_for_binomial(theta, n, H0):
    beta_theta = 0.
    for i in xrange(H0, n+1):
        beta_theta += comb(n, i) * np.power(theta, i) * np.power(1-theta, n - i)

    return beta_theta 


'''
在sigma已知的情况下，检验mu
H0: mu <= H0
H1: mu > H0
alpha = 0.05
'''
def power_function_for_norm(X_bar, n, H0):
    sigma = 1
    beta_mu = 0
    z = (X_bar - H0) / ( 1. / np.sqrt(n * 1.))
    return norm.cdf(z)

    

    return None


if __name__ == "__main__":
    n = 5
    thetas = np.linspace(0, 1, 100)
    pw_vals = np.zeros(len(thetas))
    
    H0 = 3 
    for i in xrange(len(thetas)):
        pw_vals[i] = power_function_for_binomial(thetas[i], n, H0)
    plt.plot(thetas, pw_vals, color='g')

    H0 = 5
    for i in xrange(len(thetas)):
        pw_vals[i] = power_function_for_binomial(thetas[i], n, H0)
    plt.plot(thetas, pw_vals, color='k')

    plt.savefig('images/binomial_power_function.png', format='png')

    plt.clf()

    n = 10
    X_bars = np.linspace(-2, 2, 200)
    pw_vals = np.zeros(len(X_bars))

    H0 = -1
    for i in xrange(len(X_bars)):
        pw_vals[i] = power_function_for_norm(X_bars[i], n, H0)
    plt.plot(X_bars, pw_vals, color='g')

    H0 = 0
    for i in xrange(len(X_bars)):
        pw_vals[i] = power_function_for_norm(X_bars[i], n, H0)
    plt.plot(X_bars, pw_vals, color='b')

    H0 = 1
    for i in xrange(len(X_bars)):
        pw_vals[i] = power_function_for_norm(X_bars[i], n, H0)
    plt.plot(X_bars, pw_vals, color='k')

    plt.savefig('images/norm_power_function.png', format='png')
