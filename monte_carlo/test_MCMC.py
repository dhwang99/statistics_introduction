#encoding: utf8

import numpy as np
import pdb

'''

1. 通过 markov chain 转移 模拟生成符合 pi分布的样本
    示例共有3个状态，0,1,2, 随机变量依矩阵P进行状态转移
    算法如下：
    1. 起始 随机变量X在状态0上
    2. 沿对应的转移概率进行转移：用均匀分布生成 随机数，找到对应状态，x跳到该状态上，同时对应状态样本的数量+1
    3. 转移N次后，计算转移过程中生成的各状态对应的样本量，即为统计分布

    注: 
    1. 开始时P矩阵给错了，导致sample算出的结果和pi不一致
    2. 转移次数增加后，样本方差变小

2. Metropolis-Hastings 采样
    test_MCMC
    用一个已知的Q矩阵模拟生成符合p的样本

3. 另一个算法是：(test_MCMC_2)
    1). 等概率的选下一个状态
    2). 均匀分布生成随机数，如果随机数 > 当前状态接受概率，则接收，否则拒绝
    这个还未证明(就是接收、拒绝算法?)
'''

P = np.array([[0.65, 0.28, 0.07], 
              [0.15, 0.67, 0.18], 
              [0.12, 0.36, 0.52]])

pi = np.array([0.286, 0.489, 0.225])


def test_direct(N):
    x = 0   #开始在第一个状态上
    x_samples = np.zeros(3)
    for i in xrange(N):
        Pi = P[x]
        u = np.random.random()
        
        pr = 0
        for j in xrange(3):
            pr += Pi[j]
            if u <= pr:
                x = j
                if i > 100:
                    x_samples[x] += 1
                break
    
    x_total = x_samples.sum() * 1.0
    x_samples = x_samples / x_total
    
    print "pi:", pi
    print "samples distribute ~ pi:",  x_samples

'''
Metropolis-Hastings 采样

Q: 非均匀分布的状态转移矩阵
alpha(y,xi) = min(p(y)Q(y,xi)/p(xi)Q(xi, y), 1) = min(p(y)/p(xi), 1)
alpha(y,xi) = p(y)Q(y,xi) = 1/3 * p(y)
'''
def test_MCMC(N):
    x = 0   #开始在第一个状态上
    x_samples = np.zeros(3)
    q = np.array([[0.1, 0.3, 0.6], [0.1, 0.3, 0.6], [0.1, 0.3, 0.6]])
    q = np.array([[1/3., 1/3., 1/3.], [1/3., 1/3., 1/3.],[1/3., 1/3., 1/3.]])
    q = np.array([[0.2, 0.4, 0.4], [0.4, 0.4, 0.2], [0.4, 0.2, 0.4]])

    for i in xrange(N):
        u1 = np.random.random()  #q(x,y) = 0.1, 0.3, 0.6
        can = 0
        xq = q[x, can]
        while True:
            if xq > u1:
                break
            can += 1
            xq += q[x,can]

        al = pi[can]*q[can, x]   # p(y) * Q(y, xi)
        al = pi[can]*q[can, x]/(pi[x] * q[x, can])   # p(y)*Q(y,xi)/p(xi)*Q(xi,y)

        u = np.random.random()
        if u < al:
            x = can  # 转移了. 可能会转移给自己
            #x_samples[x] += 1
        x_samples[x] += 1


    x_total = x_samples.sum() * 1.0
    x_samples = x_samples / x_total
    
    print "pi:", pi
    print "N: %s; total sample: %s; \nsamples distribute ~ pi: %s" %(N, x_total, x_samples)

'''
Q为均匀分布的转移矩阵
'''
def test_MCMC_NO_QArray(N):
    x = 0   #开始在第一个状态上
    x_samples = np.zeros(3)

    for i in xrange(N):
        can = np.random.randint(3)   #Q为均匀分布。每个转移链上，向其它状态转移的概率相同
        u = np.random.random()
        #al = pi[can]/pi[x]   # p(y)*Q(y,xi)/p(xi)*Q(xi,y)
        al = pi[can]*1/3.   # p(y) * Q(y, xi)
        if u < al:
            x = can  # 转移了. 可能会转移给自己
            x_samples[x] += 1 
        #x_samples[x] += 1 

    x_total = x_samples.sum() * 1.0
    x_samples = x_samples / x_total
    
    print "pi:", pi
    print "N: %s; total sample: %s; \nsamples distribute ~ pi: %s" %(N, x_total, x_samples)


'''
这个需要推导一下
1. 先等概率的采集某个状态下的样本
2. 再生成一个随机值u，转移概率 > u, 则接受这次转移
'''
def test_MCMC_2(N):
    x = 0   #开始在第一个状态上
    x_samples = np.zeros(3)
    for i in xrange(N):
        Pi = P[x]
        id = np.random.randint(0, 3)  #q(x,y) = 1/3
        u = np.random.random()
        if u < Pi[id]:
            x = id
            if i > 100:
                x_samples[x] += 1
    
    x_total = x_samples.sum() * 1.0
    
    x_samples = x_samples / x_total
    
    print "pi:", pi
    print "samples distribute ~ pi:",  x_samples


if __name__ == "__main__":
    N = 10000
    N = 10000

    print "Test direct:"
    test_direct(N)

    print "\nTest Metropolis :"
    test_MCMC(N)

    print "\nTest Metropolis no Q:"
    test_MCMC_NO_QArray(N)
