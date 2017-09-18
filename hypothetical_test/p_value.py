#encoding: utf8

import numpy as np
from scipy.misc  import comb
from scipy.stats import norm
import matplotlib.pyplot as plt

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
    
    print "mu0: %.4f; alpha: %.4f; c: %.4f; x_bar:%.4f p_val: %.4f" % (mu0, alpha, c, c, p_val) 
