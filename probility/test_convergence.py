#encoding: utf8

import numpy as np
import matplotlib.pyplot as plt
import pdb

'''
uniform random function
'''
def distribute_funU():
    mu = (1. + 0.)/2.0  #p = (a+b)/2
    var = (1 + 1*0 + 0)/12.  #v = (a^2 + ab + b^2)/12    
    def f():
        return np.random.random(1)[0]

    return mu, var, f

def distribute_funBinomial():
    n = 10
    p = .3    #更符合正态分布一些。取0.1之类的，就用 poisson之类的来模拟吧
    mu = n*p
    var = n* p*(1-p)
    def f():
        return np.random.binomial(n, p, 1)

    return mu, var, f

'''
f(x; beta) = 1/beta*exp(-x/beta)
F(x; beta) = 1 - exp(-x/beta) , x>0
mu = beta
Var = beta^2
'''
def distribute_funExp():
    beta = 5.
    mu = beta 
    var = beta ** 2

    def f():
        return np.random.exponential(beta)

    return mu, var, f 

'''
概率收敛：

Weak Law of Large Numbers(WLLN), 弱大数定理， 独立同分布

Xn_bar = sum(Xi)/n --> mu of X, 
limit (P(Xn_bar - mu_of_X) < epsilon) = 0, as n --> infty
Xn_bar --> mu_of_X： 样本均值依概率趋于mu_of_X

S2 = sum(Xi - Xn_bar)/(n-1)
limit (P(S2 - sigma2_of_X) < epsilon) = 0, as n --> infty
S2 --> sigma^2 of X, 样本方差依概率收敛于 X的方差

从试验结果看，随n增大，Xn和mu_X更近. 方差的偏离度要大于均值的偏离度。和平方有关
'''
def test_WLLN(n, dis_f):
    P_X,V_X, rf = dis_f()
    
    sample_count = 100
    Xn_samples = np.zeros(sample_count)
    S2_samples = np.zeros(sample_count)

    for i_xn in xrange(sample_count):
        Xi_samples = np.zeros(n)
        for i in xrange(n):
            xi = rf()                   #采样的一个结果，用小x表示
            Xi_samples[i] = xi        
        xn_bar = Xi_samples.mean()      #Xn_bar的一个采样结果，用小xn_bar表示
        s2 = np.sum(map(lambda x:(x - xn_bar)**2, Xi_samples)) / (n - 1)

        Xn_samples[i_xn] = xn_bar
        S2_samples[i_xn] = s2
    
    s1 = "P_X, Xn_mean, Xn_var: %.4f %.4f %.4f %.4f%%" % (P_X, Xn_samples.mean(), Xn_samples.var(), (P_X - Xn_samples.mean())/P_X * 100)
    s2 = "V_X, S2_mean, S2_var: %.4f %.4f %.4f %.4f%%" % (V_X, S2_samples.mean(), S2_samples.var(), (V_X - S2_samples.mean())/V_X * 100)

    print s1 + "\t"+s2

'''
分布收敛:

Central Limit Theorem, 中心极限定理

Xn_bar 近似服从期望为mu, 方差为 sigma^2/n 的正态分布 
Xn_bar ~= N(mu, sigma^2/n)
Zn = sqrt(n) *(Xn_bar - mu)/sigma, Zn ~= Z 

这个不太好弄。还是画图吧. 统一转为标准正态分布的图
以区间内归一化个数画折线图，然后画标准图。对比看结果

从图上可以比较明显看出，随X_n_bar的样本数增大，X_n_bar的方差在变小
'''

def test_CLT(n, dis_f, normed, color):
    mu_X,v_X, rf = dis_f()
    
    sample_count = 100
    sample_count = 1000
    Xn_samples = np.zeros(sample_count)
    S2_samples = np.zeros(sample_count)

    for i_xn in xrange(sample_count):
        Xi_samples = np.zeros(n)
        for i in xrange(n):
            xi = rf()                   #采样的一个结果，用小x表示
            Xi_samples[i] = xi        
        xn_bar = Xi_samples.mean()      #Xn_bar的一个采样结果，用小xn_bar表示
        s2 = v_X
        if n > 1:
            s2 = np.sum(map(lambda x:(x - xn_bar)**2, Xi_samples)) / (n - 1)


        Xn_samples[i_xn] = xn_bar
        S2_samples[i_xn] = s2
    
    #通常拿不到 真实分布的mu, sigma, 于是用它的逼近值代替
    X_bar_avg = Xn_samples.mean()
    S2_avg = S2_samples.mean()
    if normed:
        Zn_s = map(lambda x:np.sqrt(n) * (x - X_bar_avg)/np.sqrt(S2_avg), Xn_samples)
        #本code因为有真值，于是用真值代替了
        Zn_s = map(lambda x:np.sqrt(n) * (x - mu_X)/np.sqrt(v_X), Xn_samples)
    else:
        Zn_s = Xn_samples

    Zn_s.sort()
    #开始分区间数数
    group_count = int(np.sqrt(sample_count) + 0.999)  #区间数取上界
    bins = np.linspace(Zn_s[0], Zn_s[-1], group_count)
    width = (Zn_s[-1] - Zn_s[0]) / group_count
    hist = np.histogram(Zn_s, bins, normed=True)[0]

    plt.plot(bins[1:], hist, color=color)

    return None

if __name__ == "__main__":
    print "test WLLN for uniform sample:"
    for i in xrange(20):
        test_WLLN(100, distribute_funU)

    print "\n"

    print "test WLLN for Exp sample:"
    for i in xrange(20):
        test_WLLN(100, distribute_funExp)

    print "\n"
    print "test WLLN for binomial sample:"

    for i in xrange(20):
        test_WLLN(100, distribute_funBinomial)

    colors = ['k', 'g', 'y', 'b', 'r', 'm', 'y']
    plt.clf()
    n_list = [1, 3, 10, 50, 100]
    normed = False 
    id = 0
    for i in n_list:
        color = colors[id % len(colors)]
        test_CLT(i, distribute_funU, normed = normed, color=color)
        id += 1

    plt.savefig('images/convergence/convergence_uniform.png', format='png')

    plt.clf()
    id = 0
    for i in n_list:
        color = colors[id % len(colors)]
        test_CLT(i, distribute_funExp, normed = normed, color=color)
        id += 1

    plt.savefig('images/convergence/convergence_exp.png', format='png')

    plt.clf()
    id = 0
    for i in n_list:
        color = colors[id % len(colors)]
        test_CLT(i, distribute_funBinomial, normed = normed, color=color)
        id += 1

    plt.savefig('images/convergence/convergence_binomial.png', format='png')

    plt.clf()
    normed = True 
    id = 0
    for i in n_list:
        color = colors[id % len(colors)]
        test_CLT(i, distribute_funU, normed = normed, color=color)
        id += 1

    plt.savefig('images/convergence/convergence_uniform_normed.png', format='png')

    plt.clf()
    id = 0
    for i in n_list:
        color = colors[id % len(colors)]
        test_CLT(i, distribute_funExp, normed = normed, color=color)
        id += 1

    plt.savefig('images/convergence/convergence_exp_normed.png', format='png')

    plt.clf()
    id = 0
    for i in n_list:
        color = colors[id % len(colors)]
        test_CLT(i, distribute_funExp, normed = normed, color=color)
        id += 1

    plt.savefig('images/convergence/convergence_exp_normed.png', format='png')

    plt.clf()
    id = 0
    for i in n_list:
        color = colors[id % len(colors)]
        test_CLT(i, distribute_funBinomial, normed = normed, color=color)
        id += 1

    plt.savefig('images/convergence/convergence_binomial_normed.png', format='png')
