#encoding: utf8

import numpy as np
import matplotlib.pyplot as plt

import pdb

def sdnorm(z):
    """
    Standard normal pdf (Probability Density Function)
    """
    return np.exp(-z*z/2.)/np.sqrt(2*np.pi)

n = 10000
alpha = 1
x = 0.
vec = []
vec.append(x)
innov = np.random.uniform(-alpha,alpha,n) #random inovation, uniform proposal distribution
for i in xrange(1,n):
    can = x + innov[i] #candidate
    aprob = min([1.,sdnorm(can)/sdnorm(x)]) #acceptance probability
    u = np.random.uniform(0,1)
    if u < aprob:
        x = can
        vec.append(x)

#plotting the results:
#theoretical curve
x = np.arange(-3,3,.1)
y = sdnorm(x)
plt.subplot(211)
plt.title('Metropolis-Hastings')
plt.plot(vec)
plt.subplot(212)

plt.hist(vec, bins=30,normed=1)
plt.plot(x,y,'ro')
plt.ylabel('Frequency')
plt.xlabel('x')
plt.legend(('PDF','Samples'))
#show()
plt.savefig('images/test_matri.png', format='png')
