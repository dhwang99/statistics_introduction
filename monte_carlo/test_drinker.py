
#encoding: utf8

import numpy as np
#import matplotlib.pyplot as plt
from scipy.stats import norm

import pdb

'''
markov链的例子: 一个醉汉在一个5块的方格里走，经70步后，其所在位置的概率为：

x_71 rst:[ 0.09090909  0.27272727  0.27272727  0.27272727  0.09090909]
且是个稳定的概率, 这个就是 \pi, \pi * P = \pi 

15步后就很近似了:
x_15 rst:[ 0.09354559  0.27874945  0.27274217  0.2666661   0.08829669]

P^n, n >= 70, 也进入稳态 

如果把P的第5行改为[0 0 0 0 1], 则系统进入吸收态，出不来了
'''

P = np.array([[0, 1., 0, 0, 0],
    [1./3, 1./3, 1./3, 0, 0],
    [0, 1./3, 1./3, 1./3, 0],
    [0, 0, 1./3, 1./3, 1./3],
    [0, 0, 0, 1., 0]])


x0 = np.array([1, 0, 0, 0, 0.])

x_lst = []
x_lst.append(x0)
x_i = x0
print "x_0 rst:%s" % (x_i)
for id in xrange(100):
    x_i = np.dot(x_i, P)
    x_lst.append(x_i)
    print "x_%s rst:%s" % (id, x_i)

    if id > 75:
        
        print "pi*Pij vs pj * pji:"
        for i in xrange(4):
            j = i+1
            pi = x_i[i]
            pj = x_i[j]
            
            print "pi_%s*P%s_%s VS pi_%s*P%s_%s: %.5f %.5f" % (i, i, j, j, j, i, pi*P[i,j], pj*P[j,i])

PN = P
for i in xrange(100):
    PN = np.dot(PN, P)

    if i % 10 == 0:
        print "PN, ", i
        print PN

