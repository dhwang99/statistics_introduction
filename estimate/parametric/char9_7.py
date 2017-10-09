#encoding: utf8

import numpy as np
import pdb

'''
X1 ~ Binomial(n1, p1)
X2 ~ Binomial(n2, p2)
n1=n2=200, X1=160, X2=148
phi = g(p1, p2) = p1 - p2

1. find  mle phi_hat for phi
2. find fisher information matrix I(p1, p2)
3. use the multiparameter delta method to find asymptotic standard error
4. find an approximate 90% confidence interval for phi using:
    1) the delta method 
    2) the parameter bootstrap 

f(X;p1,p2) = f(X1;p1)*f(X2;p2) = p1^X1 * (1-p1)^(n1-X1) * p2^X2 * (1-p2)^(n2-X2)
logf = X1*log(p1) + (n1-X1)log(1-p1) + X2*log(p2) + (n2-X2)log(1-p2)
s(p1,p2) = nabla logf = [X1/p1 - (n1-X1)/(1-p1),     X2/p2 - (n2-X2)/(1-p2)]
nabla s = [-X1/p1^2 - (n1-X1)/(1-p1)^2 , 0; 0, -X2/p2^2 - (n2-X2)/(1-p2)^2]
I(p1, p2) = -E(nabla s)
          = [E(X1)/p1^2 + E(n1 - X1)/(1-p1)^2, 0;0, E(X2)/p2^2 - E(n2-X2)/(1-p2)^2]
          = [n1*p1/p1^2 + (n1 - n1*p1)/(1-p1)^2, 0;0, n2*p2/p2^2 - (n2-n2*p2)/(1-p2)^2]
          = [n1/p1 + n1/(1-p1), 0; 0, n2/p2 + n2/(1-p2)]

J = I.inv = [p1(1-p1)/n1 0; 0, p2(1-p2)/n2]

delta method:
    nabla g = [1;-1]
    var_hat(phi_hat) ~=  g'.T * J_hat * g' = p1_hat(1-p1_hat)/n1 + p2_hat(1-p2_hat)/n2 

'''

n1 = 200
n2 = 200
X1 = 160
X2 = 148

#mle for p1, p2
p1_hat = X1*1./n1
p2_hat = X2*1./n2

#mle for phi, mle 同变性
phi_hat = p1_hat - p2_hat 

# delta method
var_phi_hat = p1_hat*(1-p1_hat)/n1 + p2_hat * (1-p2_hat)/n2
std_phi_hat = np.sqrt(var_phi_hat)
cf = [phi_hat - 1.96 * std_phi_hat, phi_hat + 1.96 * std_phi_hat]

print "mle & delta: phi_hat:%.4f, confidence interval: %.4f, %.4f" % (var_phi_hat, cf[0], cf[1]) 

#bootstrap method
B = 1000
B_samples = np.zeros(B)
for b in xrange(B):
    p1_samples = np.random.binomial(1, p1_hat, n1)
    p2_samples = np.random.binomial(1, p2_hat, n2)

    B_samples[b] = p1_samples.sum()*1./n1 - p2_samples.sum()*1./n2 

var_hat_bootstrap = B_samples.std()
cf = [phi_hat - 1.96 * var_hat_bootstrap, phi_hat + 1.96 * var_hat_bootstrap]
print "confidence interval by bootstrap: %.4f, %.4f" % (cf[0], cf[1])






