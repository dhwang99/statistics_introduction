#encoding: utf8

import numpy as np
import pdb

'''
极大似然估计.

L(x;theta) = max PI(p(xi;theta)) = max(sum(log(p(xi;theta))))

'''

'''
1. likelihood function: 
   f(x; theta) = 1/theta*exp(-x/theta), x>0
   L(x; theta) = PI(f)

2. log likelihood function:
   log f = -log(theta) - x/theta
   l(x;theta) = sum(log f(xi; theta)) = -n * log(theta) - sum(xi)/theta 

   注：二阶导 不一定小于0

3. deriv:
   l(x;theta) = -n/theta + sum(xi)/theta^2 = 0

   theta = sum(xi)/n
'''

def ML_for_Exp(n, theta):
    samples = np.random.exponential(theta, n)
    theta_head = samples.mean()

    return (theta_head)


'''
1. 
  L(x;p) = PI(p^xi * (1-p)^(1-xi))
2. 
  l(x;p) = sum(xi * logp + (1-xi) * log(1-p))
         = logp*sum(xi) +  log(1-p)*sum(1-xi)  , xi为0或1
         = logp * sum_one +  log(1-p) * sum_zero 
3. deriv:
  partial l /partial p = sum_one/p - sum_zero/(1-p) = 0
  (1-p)sum_one - p*sum_zero = 0
  p = sum_one / (sum_one + sum_zero) = sum_one / total_samples
'''
def ML_for_Bernolli(n, p):
    samples = np.random.binomial(1, p, n)
    sum_one = samples.sum()
    p_head = sum_one * 1. / n

    return (p_head)

'''
1. 
   f(xi;mu, sigma) = 1./(sqrt(2pi)*sigma) * exp(-(xi-mu)^2/(2*sigma^2)
   L = PI(f(xi; mu, sigma)
2. 
   l(theta) = sum(log(1./(sqrt(2pi)*sigma)) - sum((xi-mu)^2/(2*sigma^2)
            = sum(log(1/sqrt(2pi) -1/2*log(sigma^2) - sum((xi-mu)^2/(2*sigma^2)

3. deriv:
    sum(xi - mu) / (2*sigma^2) = 0
    mu = sum(xi)/n

    partial(l)/partial(sigma^2) = -n/2*1/(sigma^2) + sum((xi-mu)^2) / 2*sigma^4 = 0
    sigma^2 = sum(xi-mu)^2/n   #非无偏估计
'''
def ML_for_Norm_mu(n, mu, sigma):
    samples = np.random.normal(mu, sigma, n)
    mu_head = samples.mean()

    v = samples - mu_head
    v2 = np.dot(v, v) / n
    v2 = np.sqrt(v2)

    return (mu_head, v2)

def ML_test(test_count, l_fun, sample_count, *param):
    es_num = len(param)
    errors = np.zeros((test_count, es_num))
    thetas = np.zeros((test_count, es_num))

    for i in xrange(test_count):
        thetas_head = l_fun(sample_count, *param)
        errors[i, :] = np.array(param) - thetas_head
        thetas[i, :] = thetas_head

    for i in xrange(es_num):
        print "real param (%s,%s): %s; estimate: %s; var: %s" % (es_num, i, param[i], thetas[:,i].mean(), thetas[:,i].var())

if __name__ == "__main__":
    test_count = 10
    sample_count = 500

    print 'maximum likelihood for exp distribute samples:'
    ML_test(test_count, ML_for_Exp, sample_count, 5)

    print '\nmaximum likelihood for Bernolli distribute samples:'
    ML_test(test_count, ML_for_Bernolli, sample_count, 0.2)
    
    print '\nmaximum likelihood for Normal distribute samples: Mu:'
    ML_test(test_count, ML_for_Norm_mu, sample_count, 2, 3)
    
