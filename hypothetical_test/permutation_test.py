#encoding: utf8

import numpy as np
import pdb
import matplotlib.pyplot as plt

'''
H0: 
假设 X, Y是一样的，则T_star为(0, N!)的均匀分布

置换检验，近似求p值
'''

def mouse_test(): 
    deal_group = np.array([94, 197, 16, 38, 99, 141, 23])
    ctrl_group = np.array([52, 104, 146, 10, 51, 30, 40, 27, 46])

    dn = len(deal_group)
    cn = len(ctrl_group)

    X = np.hstack((deal_group, ctrl_group))
    T_abs = deal_group.mean() - ctrl_group.mean()

    N = 1000
    T_samples = np.zeros(N)

    for i in xrange(N):
        X_star = np.copy(X)
        np.random.shuffle(X_star)

        T_samples[i] = X_star[:dn].mean() - X_star[dn:].mean()


    T_samples.sort()
    indicate = np.where(T_samples >= T_abs)
    I_count = len(indicate[0])

    p = I_count * 1./N
    print 'X_bar:%.4f; Y_bar:%.4f; p value for permutation: %.4f' % (deal_group.mean(), ctrl_group.mean(), p) 

    group_count = int(np.sqrt(N) + 0.99)
    plt.hist(T_samples, bins = group_count, color='r', normed=True, alpha=1.0, histtype='step')
    plt.plot([T_abs, T_abs], [0, 0.01], color='b')
    plt.savefig('images/permulation.png', format='png')

'''
用中值比较. 也可以用均值
'''
def gene_check_test():
    g1 = np.array([230,-1350,-1580,-400,-760])
    g2 = np.array([970,110,-50,-100,-200])
    
    g1n = len(g1)
    T_abs = np.abs(np.median(g1) - np.median(g2)) 
    X = np.hstack((g1, g2))

    B = 1000
    T_samples = np.zeros(B)
    for b in xrange(B):
        X_star = np.copy(X)
        np.random.shuffle(X_star)
        T_samples[b] = np.median(X_star[:g1n]) - np.median(X_star[g1n:])

    indicate = np.where(T_samples > T_abs)
    I_count = len(indicate[0])
    p = I_count * 1. / B

    print "T_abs: %.4f; p: %.4f" % (T_abs, p)



if __name__ == '__main__':
    mouse_test()

    gene_check_test()
