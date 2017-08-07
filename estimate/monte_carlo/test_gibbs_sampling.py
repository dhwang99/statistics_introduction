#encoding: utf8

import numpy as np
import pdb


'''
试验二维的情况
p(x0, y0) = 0.05
p(x0, y1) = 0.10
p(x0, y2) = 0.15
p(x1, y0) = 0.10
p(x1, y1) = 0.15
p(x1, y2) = 0.20
p(x2, y0) = 0.10
p(x2, y1) = 0.10
p(x2, y2) = 0.05

1. 先固定y轴，采集x
2. 固定x轴，采集y 
'''

pi = np.array([[0.05, 0.10, 0.15],
              [0.10, 0.15, 0.20],
              [0.10, 0.10, 0.05]])

def test_gibbs(N):
    samples = np.zeros(pi.shape)
    x0 = np.array([0,0])
    
    x = x0   #采样点的坐标
    for i in xrange(N):
        px0_given_x1 = pi[:, x[1]]/(pi[:, x[1]].sum())
        #pdb.set_trace()
        t = 0.
        u = np.random.random()
        for id in xrange(pi.shape[0]):
            t += px0_given_x1[id]
            if u <= t:
                x[0] = id
                samples[x[0], x[1]] += 1
                break

         
        px1_given_x0 = pi[x[0], :]/pi[x[0], :].sum()
        #pdb.set_trace()
        u = np.random.random()
        t = 0.
        for id in xrange(pi.shape[1]):
            t += px1_given_x0[id]
            #pdb.set_trace()
            if u <= t:
                x[1] = id
                samples[x[0], x[1]] += 1
                break
            
    total = samples.sum() * 1.
    
    print "PI:\n", pi
    print "gibbs samples:\n", samples / total

if __name__ == "__main__":
    N = 10000
    test_gibbs(N)
