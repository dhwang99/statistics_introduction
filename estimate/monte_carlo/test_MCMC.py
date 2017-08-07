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
        hit = False
        for j in xrange(3):
            pr += Pi[j]
            if u <= pr:
                x = j
                hit = True
                if i > 100:
                    x_samples[x] += 1
                break
    
        if hit == False:
            print "not Hit. U:", u
    
       
    
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
    q = np.array([[1/3., 1/3., 1/3.], [1/3., 1/3., 1/3.],[1/3., 1/3., 1/3.]])
    q = np.array([[0.1, 0.3, 0.6], [0.1, 0.3, 0.6], [0.1, 0.3, 0.6]])

    for i in xrange(N):
        u1 = np.random.random()  #q(x,y) = 0.1, 0.3, 0.6
        id = 0
        t = q[x, id]
        while True:
            if u1 < t:
                break
            id += 1
            t += q[x, id]

        al = pi[id]*q[id, x]   # p(y) * Q(y, xi)
        al = pi[id]*q[id, x]/(pi[x] * q[x, id])   # p(y)*Q(y,xi)/p(xi)*Q(xi,y)
        u = np.random.random()
        if u < al:
            x = id  # 转移了. 可能会转移给自己
        else:
            x = x   # 未发生转移。 这次采样转移失败, 状态不变。这个一定要加上，要不会出问题. 但这个是错的

        if i > 100:
            x_samples[x] += 1 

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
    N = 100000
    N = 10000

    print "Test direct:"
    test_direct(N)

    print "\nTest AB:"
    test_MCMC(N)
