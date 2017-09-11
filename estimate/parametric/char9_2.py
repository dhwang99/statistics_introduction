#encoding: utf8

import numpy as np
import pdb

'''
moment:
    a1 = (b+a)/2, a1 = 1/n sum(Xi)
     
    a2 = E(X^2) = (b^2 + ab + a^2)/3, a2 = 1/n sum(Xi^2)
    or 
    a2 = V(X) + E^2(X) 
    a2 - a1 = V(X) = (b-a)^2/12 

    (b-a)^2/12 = a2 - a1^2
    (b - 2*es_a1 + b) = sqrt(12*(a2 + a1^2))
    hat_b = a1 + sqrt(3*(a2+ a1^2))
    hat_a = a1 - sqrt(3*(a2+ a1^2))

    tau_hat = 1/n * sum(Xi)

    mse: find analytically
    
    bias(tau) = 0
    var(tau) = 1/n var(X)
    mse = var(tau) + bias(tau)**2 = var(tau)

mle:
 because:
   L(a,b) = \Pi (1/(b - a)^n, min(Xi) >= a, max(Xi) <= b
   otherwise: L(a,b) = 0 (因为连乘，Xi <a或Xi>b时，概率为0， 连乘为0)
 therefore: 
   a = min(Xi), b = max(Xi), L取到最大值

   tau_hat = (a_hat + b_hat)/2

 mse: simulate by parameter bootstrap, and tau alternated by tau_hat (tau_hat --P--> tau)
     
'''
def ML_for_uniform(n, a, b):
    tau = (b + a)/2.
    print "real value: a: %.4f; b: %.4f; tau: %.4f" % (a, b, tau)
    samples = np.random.random(size=n) * (b - a) + a
    a1 = samples.mean()
    a2 = np.dot(samples, samples)/n
    hat_b = a1 + np.sqrt(3*(a2- a1**2))
    hat_a = a1 - np.sqrt(3*(a2- a1**2))

    hat_tau = (hat_b + hat_a)/2

    print "moment: a:%.4f; b:%.4f, tau: %.4f" % (hat_a, hat_b, hat_tau)
    
    #mle
    hat_a = np.min(samples)
    hat_b = np.max(samples)
    
    # \tau = \int xdF(x), estimate hat_tau
    # \tau = g(a,b) = (b-a)/2, 由mle估计的同变性，hat_tau = (hat_b - hat_a)/2
    hat_tau = (hat_b + hat_a)/2
    mle_mse = 0.0
    #mse of hat_tau by simulation. parameter bootstrap
    B = 1000 
    B_samples_for_tau = np.zeros(B)
    for bi in xrange(B):
        b_samples = np.random.random(n) * (hat_b - hat_a) + hat_a
        '''
        注意，模拟方法不是求mean, 而是下述算法
        B_samples_for_tau[b] = b_samples.mean()
        '''
        B_samples_for_tau[bi] = (b_samples.min() + b_samples.max())/2
    
    bias_tau = B_samples_for_tau.mean() - hat_tau #why is not tau? because this is simulation. use hat_tau altinatively 
    mle_mse = B_samples_for_tau.var() + bias_tau * bias_tau

    print "mle: a:%.4f; b:%.4f, tau: %.4f; mse: %.4f" % (hat_a, hat_b, hat_tau, mle_mse)

    #nonparametric plugin-estimator \sim r(x)dF(x) = 1/n sum(r(xi))
    plugin_tau = samples.mean()  # 1/n*sum(r(Xi)) = 1/n * sum(Xi)
    plugin_mse = 0.0

    #mse of sim_tau analytically.
    #bar_X: 1/n Var(X)
    plugin_mse_analytically = 1./n * (b -a)*(b-a)/12.
    
    print "plugin estimator: tau: %.4f; mse: %.4f" % (plugin_tau, plugin_mse_analytically)

if __name__ == "__main__":
    a = 1
    b = 3
    n = 10
    ML_for_uniform(n, a, b)
