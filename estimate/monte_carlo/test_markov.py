#encoding: utf8

import numpy as np

'''
通过 markov chain 转移 模拟生成符合 pi分布的样本

示例共有3个状态，0,1,2, 随机变量依矩阵P进行状态转移

算法如下：
1. 起始 随机变量X在状态0上
2. 沿对应的转移概率进行转移：用均匀分布生成 随机数，找到对应状态，x跳到该状态上，同时对应状态样本的数量+1
3. 转移N次后，计算转移过程中生成的各状态对应的样本量，即为统计分布

注: 
1. 开始时P矩阵给错了，导致sample算出的结果和pi不一致
2. 转移次数增加后，样本方差变小

另一个算法是：(testAB)
1. 等概率的选下一个状态
2. 均匀分布生成随机数，如果随机数 > 当前状态接受概率，则接收，否则拒绝
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

def test_MCMC(N):
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
    test_AB(N)
