#encoding: utf8

import numpy as np

'''
x ~ uniform(0, 0:l/2)
y ~ uniform(0, pi/2)
g(X, Y)) = 1, X <= a/2 * sin(Y); 0, otherwise

pi = 2l/(aP(A))
'''
def test_pufun(l, a, N): 
    hc = 0
    for i in xrange(N):
        x = np.random.random() * a/2.
        theta = np.random.random() * np.pi/2.
        if l/2.*np.sin(theta) >= x:
            hc += 1

    pi = 2 * l / (a * hc * 1./N)
    print "intify result for %s:%s", N, pi

a=10.
l=2.

test_pufun(l, a,  100)
test_pufun(l, a,  1000)
test_pufun(l, a,  10000)
test_pufun(l, a,  100000)


        



    
