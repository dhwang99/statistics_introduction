#encoding: utf8

import numpy as np
import matplotlib.pyplot as plt

'''
p(z), the source distribution; 
q(z):  
accept=Mq(z)/p(z)
'''

def triangle_distribute(x, param):
    a,b,c=param
    h = 2./(b - a)
    pdf, cdf = 0., 0.

    if x < a:
        return np.array([0,0])

    if x > b:
        return np.array([0, 1])
    
    if x <= c:
        pdf = h * (x-a)/(c-a)
        cdf = (x-a) * pdf/2.
    else:
        pdf = h*(b-x)/(b-c)
        cdf = 1 - (b-x)*pdf/2.

    return np.array([pdf, cdf])


def triangle_accept_U_sample(n, param):
    a,b,c = param
    M = 2./(b-a)  #the height of triangle
    U_samples = a + np.random.random(n) * (b - a)
    ac_list = []
    for x in U_samples:
        tpdf, tcdf = triangle_distribute(x, param)
        
        u = np.random.random(1)
        if u <= tpdf/M:  #acceptable
            ac_list.append(x)

    return ac_list  #all accepted samples

if __name__ == "__main__":
    n = 100000
    tri_param = np.array([1., 6., 3.])
    a,b,c = tri_param
    height = 2./(b - a)

    tri_acc_samples = triangle_accept_U_sample(n, tri_param)
    
    sample_count = len(tri_acc_samples)
    group_count = int(np.sqrt(sample_count) + 0.999) / 5
    bins = np.linspace(a, b, group_count) 
    hist = np.histogram(tri_acc_samples, bins, normed=True)[0]
  
    plt.plot(bins[:-1]+(bins[1]-bins[0])/2, hist, color='k')  #取中间点

    source_bins = np.array([[a, 0],[c, height],[b,0]])
    plt.plot([a,c,b], [0, height, 0], color='r')

    plt.savefig('images/triangle_accept.png', format='png')
