#encoding: utf8

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import pdb

'''

Beta 分布

'''

'''
当a,b有一个为1时，为单调函数, 两个都为1时，则为[0,1]间的均匀分布

f(x;a,b) = x^(a-1)*(1-x)^(b-1)/B(a,b)
B(a,b) = Gamma(a)Gamma(b)/Gamma(a+b-1)

E(B) = a/(a+b)
V(B) = ab/[(a+b)^2(a+b+1)]

假设我们预计运动员整个赛季的击球率大概是0.27左右，范围大概是在0.21到0.35之间。那么用贝塔分布来表示，我们可以取参数 α==81，β==219。
为什么估计为这个结果？下面的算法好象也不靠谱
B_bar = 0.27
se_hat(B) = 0.024 (这个是瞎算的。)

b = 3a
a = (3/(16 * V) - 1)/4

应该是去年打了300次，有81次中，219次没有中。于是有了这个先验
'''

gamma_values = {}


def beta_pdf(x, a, b):
    gamma_a = gamma_values[a] 
    gamma_b = gamma_values[b] 
    gamma_ab = gamma_values[a+b] 

    pdf = np.power(x, a-1) * np.power(1-x, b- 1) * gamma_ab / gamma_a / gamma_b
    
    return pdf

def draw_beta_dist(ab_list, fname):
    colors = ['r', 'b', 'k', 'g', 'm', 'c']
    ls = []
    ab_lables = map(lambda x:'%s:%s' % x, ab_list) 

    for i in range(len(ab_list)):
        a,b = ab_list[i]
        points = np.linspace(0., 1., 101)
        '''
        pdfs = map(lambda x:beta_pdf(x, a,b), points)
        l, = plt.plot(points, pdfs, color=colors[i%len(colors)])
        '''

        pdfs = map(lambda x:beta.pdf(x, a,b), points)
        l, = plt.plot(points, pdfs, color=colors[i%len(colors)])
        ls.append(l)
    

    plt.legend(handles = ls, labels = ab_lables, loc = 'best')
    plt.savefig(fname, format='png')

if __name__ == '__main__':
    ab_list = [(0.5, 0.5), (5, 2), (1, 3), (2, 2), (3,1), (2,5), (5,5)]
    fname = 'images/beta01.png'
    draw_beta_dist(ab_list, fname)
    
    plt.clf()
    ab_list = [(20, 219), (81, 219), (150, 219), (219, 219), (500, 219), (1000,219)]
    fname = 'images/beta02.png'
    draw_beta_dist(ab_list, fname)

    plt.clf()
    ab_list = [(5, 1), (5, 3), (5, 5), (5, 7), (5,10), (5, 20)]
    fname = 'images/beta03.png'
    draw_beta_dist(ab_list, fname)

    plt.clf()
    ab_list = [(9, 7), (7, 4)]
    fname = 'images/beta04.png'
    draw_beta_dist(ab_list, fname)
