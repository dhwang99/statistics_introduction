#encoding: utf8

import numpy as np
import pdb

'''
f(x) = x**3
'''
def f1_intify(bins):
    return 1./4 * ( bins[1]**4 - bins[0]**4) 

def f1(x):
    return x**3

def norm_cdf(bins):
    from scipy.stats import norm

    return norm.cdf(bins[1]) - norm.cdf(bins[0])

def norm_pdf(x):
    f = 1.0/np.sqrt(2*np.pi) * np.exp(-x**2/2)
    return f


'''
X为[a,b]上均匀分布的随机变量, X = a + U(X) * (b - a), f(X) = 1/(b-a)
g在[a,b]区间上的积分：
intify(g) = sum(g(x)) 
          = (b-a)sum(g(X) * 1/(b -a)) 
          = (b-a)*sum(g(X) * f(X)) 
          = (b-a)sum[g(X)f(X)]
          = (b-a)E[g(X)]
    E(g) ~= mu(g) = 1/n * sum(g(xi))

从试验结果看，随试验次数增加，积分结果更近真似值。如n=100000时，差别比较小

'''
def intify_sim(f, bins, n):
    n_list = bins[0] + np.random.random(n) * (bins[1] - bins[0])
    
    n_rst = map(lambda x:f(x), n_list)
    n_rst = np.array(n_rst)

    mu = 1./n * np.sum(n_rst) * (bins[1] - bins[0])

    return mu

def test_intify_sim(f, f_intify, bins, n):
    sv = intify_sim(f, bins, n) 
    rv = f_intify(bins)

    print "sim value: %s; real value: %s" % (sv, rv)

'''
monte carlo 求正态积分 Phi(x)值:
1. 取n个 正态分布的样本, xi
2.  g(xi) = 1, if xi <= x, else g(xi) = 0
Phi(x) = sum((g(xi) * phi(xi)) = E(g(xi))
mu(g) = 1/n * sum(g(xi)) 
其实就是：把采样得到样本，<= x的样本数/总样本数
'''
def test_norm_intify_sim():
    from scipy.stats import norm
    xps = [[0., norm.cdf(0)], [1, norm.cdf(1)], [2, norm.cdf(2)]]
    n = 10000
    
    for norm_x_p in xps:
        x_list = np.random.normal(0, 1, n)
        samples_count = np.where(x_list <= norm_x_p[0])[0].size
        sim_v = samples_count * 1. / n

        print "x: %s; real v: %s; sim_v: %s" %(norm_x_p[0], norm_x_p[1], sim_v)


def calcuate_pi():
    nlist = [10000,100000,1000000]
    for n in nlist:
        x1 = np.random.random(n)
        x2 = np.random.random(n)
        x = np.array([x1,x2]).T
        r2_list = np.apply_along_axis(lambda a: np.dot(a,a)/4., 1, x)
        pi_count = np.where(r2_list <= 0.25)[0].size
        print "n: %s; pi value:%s" % (n, pi_count * 4./n)

if __name__ == "__main__":
    n = 10000
    n = 100000
    n = 1000000

    print "sim for 1/x**3 intify:"
    bins = np.array([0., 2])
    test_intify_sim(f1, f1_intify, bins, n)

    bins = np.array([0., 10])
    test_intify_sim(f1, f1_intify, bins, n)

    print "\nsim for norm intify:"
    test_norm_intify_sim()

    print "\nsim for norm intify by U distribution:"
    bins = np.array([-1., 1])
    test_intify_sim(norm_pdf,  norm_cdf, bins, n)

    print "\nsim for norm intify by U distribution:"
    bins = np.array([-2., 2])
    test_intify_sim(norm_pdf,  norm_cdf, bins, n)

    print "\n calcuate PI:"
    calcuate_pi()
