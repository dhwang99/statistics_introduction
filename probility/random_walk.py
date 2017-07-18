# encoding: utf8

import numpy as np
import matplotlib.pyplot as plt

import pdb

'''
随机游走.random walk
在 股票市场 预测用得比较多

example:
   walk one unit to left by p, right by (1-p)
   Y = sum(Xi), Xi, random Variable for each step 

   f(x) = p, x=-1,  go left
          1-p, x=1, go right

   E(Xi) = -1 * p + 1*(1-p) = 1-2p 
   E(Xi^2) = 1*p + 1*(1-p) = 1
   E(Xi) ^ 2 = 1 - 4p + 4p*p
   V(Xi) = E(Xi^2) - E(Xi)^2 = 4p(1-p)

   E(Y) = n(1-2p)
   V(Y) = 4np(1-p)

从图里的结果看，当p >= 0.3时，几次试验的结果，变化都很大。不确定性太多了
可以对照 最后 100 步的图

从这个例子看，p = 0.5, 累积行为基本不可预测？
'''

'''
n: steps
p: probility

return expect, varible
'''
def EV_for_random_walk(n,p):
    E = n*(1-2*p)
    V = 4*n*p*(1-p)

    return (E,V)


def random_walk(n,p): 
    pos = 0.
    pos_list = []

    for i in xrange(1, n+1):
        rd_num = np.random.random()
        if rd_num < p:
            pos -= 1
        else:
            pos += 1

        pos_list.append(pos)

    return pos_list


def plot_random_walk(n, p, last_steps, repeat_times):
    E_list = []
    V_list = []
    std1_list = []
    std2_list = []
    steps = range(1, n+1)
    colors = ['b', 'g', 'y', 'k', 'c', 'm', 'y']

    for i in steps:
    	E,V = EV_for_random_walk(i, p)
    	E_list.append(E)
        V_list.append(V)

    std1_list = np.sqrt(V_list) + E_list
    std2_list = -np.sqrt(V_list) + E_list

    plt.clf()
    for i in xrange(repeat_times):
        pos_list = random_walk(n, p)
        color = colors[i%len(colors)]
        #pdb.set_trace()        
        plt.plot(steps[-last_steps:], pos_list[-last_steps:], color=color, lw=0.5)

    lw=2
    plt.plot(steps[-last_steps:], E_list[-last_steps:], color='b', lw=lw)
    plt.plot(steps[-last_steps:], std1_list[-last_steps:], color='r', lw=lw)
    plt.plot(steps[-last_steps:], std2_list[-last_steps:], color='r', lw=lw)

    plt.savefig('images/random_walk/laststep_%s_p%s.png' % (last_steps, p), format='png')

if __name__ == "__main__":
    for i in range(1,6):
        plot_random_walk(10000, 0.1 * i, 10000,  5)
        plot_random_walk(10000, 0.1 * i, 100,  5)

