#encoding: utf8

import numpy as np

'''
极大似然估计.

L(x;theta) = max PI(p(xi;theta)) = max(sum(log(p(xi;theta))))

uniform不能通过极大似然求解。pdf: f(x) = 1/(b - a), 没有x参数

'''

'''
1. likelyhood function: 
   f(x; theta) = 1/theta*exp(-x/theta), x>0
   L(x; theta) = PI(f)

2. log likelyhood function:
   log f = -log(theta) - x/theta
   l(x;theta) = sum(log f(xi; theta)) = -n * log(theta) - sum(xi)/theta 

   logx, 1/x都是凹函数， 则上述函数为凸函数，可以求上式最大值

3. solve:
   l(x;theta) = -n/theta + sum(xi)/theta^2 = 0

   theta = sum(xi)/n
'''
def ML_for_Exp(n, theta):
    samples = np.random.exponential(theta, n)
    theta_head = samples.mean()

    return theta_head 


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

    return p_head

'''
1. 
   f(xi;mu, sigma) = 1./(sqrt(2pi)*sigma) * exp(-(xi-mu)^2/(2*sigma^2)
   L = PI(f(xi; mu, sigma)
2. 
   l(sigma) = sum(log(1./(sqrt(2pi)*sigma)) - sum((xi-mu)^2/(2*sigma^2)

3. deriv:
    sum(xi - mu) / (2*sigma^2) = 0
    mu = sum(xi)/n
'''
def ML_for_Norm(n, mu, sigma):
    samples = np.random.normal(mu, sigma, n)
    mu_head = samples.mean()
    return mu_head

def ML_test(test_count, l_fun, sample_count, *param):
    errors = np.zeros(test_count)
    thetas = np.zeros(test_count)
    for i in xrange(test_count):
        theta_head = l_fun(sample_count, *param)
        
        errors[i] = param[0] - theta_head
        thetas[i] = theta_head

    
    print "real param: %s; estimate mean: %s; var: %s" % (param, thetas.mean(), thetas.var())

if __name__ == "__main__":
    test_count = 10
    sample_count = 500

    print 'maxminum likelyhood for exp distribute samples:'
    ML_test(test_count, ML_for_Exp, sample_count, 5)

    print ''
    print 'maxminum likelyhood for Bernolli distribute samples:'
    ML_test(test_count, ML_for_Bernolli, sample_count, 0.2)
    
    print ''
    print 'maxminum likelyhood for Normal distribute samples:'
    ML_test(test_count, ML_for_Norm, sample_count, 2, 3)
    
