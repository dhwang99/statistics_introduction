#encoding: utf8

'''
用 uniform(0,1)测试样本分布

X_bar = sum(Xi) * /n

E(X_bar) = mu. 即X_bar均值的期望是总体的均值
V(X_bar) = V(X)/n. X_bar 的方差为总体方差/n
S2 = sum(Xi-X_bar)^2/(n-1)

V(S2)是多少？也是 var(X)/(n-1)? 画了个图。但没有理论

留个问题：采样10000次，和采样100次，每次采100个样例，算出来的均值和方差，哪个方法更好？还是这种方法本身就有问题

1. 从结果看，X_bar 的方差和确实和 var(X)/n接近
'''

import matplotlib.pyplot as plt
import numpy as np
import pdb

def gen_Xbar_sample_byU(n):
    total = 0
    x_samples = np.random.random(n)
    x_bar = x_samples.mean()
    d = x_samples - x_bar
    s2 = np.dot(d, d)
    if n > 1:
        s2 = s2 / (n - 1)

    return (x_bar, s2)

if __name__ == "__main__":
    #U(0,1)的期望和方差
    M_X = (1. + 0.)/2
    V_X = (1. - 0.)**2/12.
  
    nlist = [1, 5, 25, 100, 1000]
    sample_count = 100

    x_bar_means = []
    x_bar_mu_theorem = []
    x_bar_vars = []
    x_bar_vars_theorem = []

    x_s2_vars = []
    test_nums = range(1, 101)
    for n in test_nums:
        x_bar_samples = []
        x_bar_s2 = []
        for i in xrange(1, sample_count + 1):
            #采样100次, 求mean和方差
            x_bar, s2 = gen_Xbar_sample_byU(n)
            x_bar_samples.append(x_bar)
            x_bar_s2.append(s2)
        
        #计算x_bar的均值
        x_bar_mean = np.mean(x_bar_samples)
        #求x_bar的方差
        v = x_bar_samples - np.ones(sample_count) * x_bar_mean
        x_bar_var = np.dot(v, v) / (sample_count-1)  #x_bar的样本方差, 其1/(n-1)是总体方差的无偏估计
        #求S2的方差
        v = x_bar_s2 - np.mean(x_bar_s2) * x_bar_mean
        s2_var = np.dot(v, v)/(sample_count - 1)
        #pdb.set_trace()
        v = np.var(x_bar_samples)

        x_bar_means.append(x_bar_mean)
        x_bar_mu_theorem.append(M_X)
        x_bar_vars.append(x_bar_var)
        x_bar_vars_theorem.append(V_X/n)

        x_s2_vars.append(s2_var)
   
    imgdir = 'images/sample_distribution'
    plt.clf()
    plt.plot(test_nums, x_bar_means, color='b')
    plt.plot(test_nums, x_bar_mu_theorem, color='r')
    plt.savefig(imgdir + '/x_bar_means.png', format='png')
    
    plt.clf()
    plt.plot(test_nums, x_bar_vars, color='b')
    plt.plot(test_nums, x_bar_vars_theorem, color='r')
    plt.savefig(imgdir + '/x_bar_vars.png', format='png')
    
    plt.clf()
    plt.plot(test_nums, x_s2_vars, color='b')
    #plt.plot(test_nums, x_bar_vars_theorem, color='r')
    plt.savefig(imgdir + '/x_s2_vars.png', format='png')
    
    sample_ids = range(1, sample_count + 1)
    for n in nlist:
        x_bar_samples = []
        x_bar_s2 = []
        for i in xrange(1, sample_count + 1):
            x_bar, s2 = gen_Xbar_sample_byU(n)
            x_bar_samples.append(x_bar)
            x_bar_s2.append(s2)
         
        fname = imgdir + "/x_bar_sample_n%s.png" % (n)
        plt.clf()
        plt.plot(sample_ids, x_bar_samples, color='b')
        plt.plot(sample_ids, x_bar_mu_theorem[:len(sample_ids)], color='b')
        plt.savefig(fname, format='png')
