#encoding: utf8

import numpy as np
from scipy.misc  import comb
from scipy.stats import norm

import pdb

from power_function_example import c_norm
'''
计算正态分布下的p值

p_value = 1 - Phi_inv((mu_bar - c)*np.sqrt(n)/sigma)
'''
def pvalue_for_norm(mu_bar, n, mu0, sigma):
    Z_a = (mu_bar - mu0) * np.sqrt(n*1.) / sigma
    p_val = 1 - norm.cdf(Z_a)
    return p_val

'''
example: 
X_bar = 1/n*sum(Xi) = 195.27
se_X_bar = sqrt(sum(Xi - X_bar)^2/(n-1) = 5.0
Y_bar = 1/m*sum(Yi) = 216.19
se_Y_bar = sqrt(sum(Yi-Y_bar)^2/(m-1) = 2.4

H0: delta = X - Y = 0; 
'''
def p_for_delta1():
    X_bar = 195.27
    se_X_bar = 5.0
    Y_bar = 216.19
    se_Y_bar = 2.4
    
    '''
    Wald value = (T(Xn) - T(X0))/sigma
    '''
    W = (X_bar - Y_bar) / np.sqrt(se_X_bar**2 + se_Y_bar**2)

    p_val = 2 * norm.cdf(-np.abs(W))

    print " p_val is: %.4f" % (p_val)
     

if __name__ == "__main__":
    colors = ['g', 'b', 'k']
    #test contain of samples
    mu0 = 0
    sigma = 1.
    mu0 = 0
    alpha = 0.05
    n = 10

    c = c_norm(mu0, sigma, alpha, n)
    p_val = pvalue_for_norm(c, n, mu0, sigma)
    
    print "Norm test: mu0: %.4f; alpha: %.4f; c: %.4f; x_bar:%.4f\n p_val: %.4f" % (mu0, alpha, c, c, p_val) 

    print "Wald test:"
    p_for_delta1()

