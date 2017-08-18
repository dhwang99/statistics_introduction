#encoding: utf8

import numpy as np
from scipy.stats import norm

'''
ex9
h_p1 = 1/n sum(X1i)
h_p2 = 1/n sum(X2i)

h_V1 = h_p1*(1 - h_p1)    
h_V2 = h_p2*(1 - h_p2)

theta = p1 - p2
h_theta = h_p1 - h_p2

h_V_p1 = h_V1/n1
h_V_p2 = h_V2/n2

h_se_theta = sqrt(h_V_p1 + h_V_p2) 
'''

h_p1 = 90./100
h_p2 = 85./100

h_theta = h_p1 - h_p2

#注意：这儿求的是 theta的方差估计， theta = 1/n * sum(Xi), 故而方差为1/(n**2) * (n * var(Xi)) = 1/n * p(1-p)
#这就是计算方差的公式，也比较好理解： n个IID的随机变量的均值也是随机变量，它的均值为1/nsum(Xi), 方差也是1/nVar(Xn)

h_se_theta = np.sqrt(0.9*0.1/100. + 0.85 * 0.15/100.) 

alpha = 0.2
delta = h_se_theta * norm.ppf(1-2*alpha)
print "confidence interval for alpha=%s is: %s" % (alpha, [h_se_theta - delta, h_se_theta + delta])

alpha = 0.025
delta = h_se_theta * norm.ppf(1-2*alpha)
print "confidence interval for alpha=%s is: %s" % (alpha, [h_se_theta - delta, h_se_theta + delta])

