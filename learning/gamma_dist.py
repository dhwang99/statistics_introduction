#encoding: utf8

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
import pdb

'''

Gamma 分布

'''

'''

f(x;a,b) = b^a*x^(a-1)*e^(b*x)/B(a), x \in (0, \infty)

E(Ga) =   
V(B) = 

'''

gamma_values = {}

'''
gamma(0.5) = sqrt(pi)
gamma(1) = 1

gamma(alpha) = (alpha - 1) gamma(alpha - 1)
'''
def init_gamma_values():
    gamma_values[0] = 1e10   #sim to infty
    gamma_values[0] = 10000
    gamma_values[0.5] = np.sqrt(np.pi)
    gamma_values[1] = 1
    
    for i in range(1, 1000):
        a1 = i - 0.5
        a2 = i 

        gamma_values[a1+1] = a1 * gamma_values[a1]
        gamma_values[a2+1] = a2 * gamma_values[a2]

init_gamma_values()

def gamma_pdf(x, a, b):

    gamma_a = gamma_values[a]

    pdf = np.power(a, b) * np.power(x, a-1) * np.exp(-b*x) / gamma_a
    
    return pdf

def draw_gamma_dist(ab_list, fname, scale):
    colors = ['r', 'b', 'k', 'g', 'm', 'c']
    ls = []
    ab_lables = map(lambda x:'%s:%s' % x, ab_list) 

    for i in range(len(ab_list)):
        a,b = ab_list[i]
        points = np.linspace(0., scale, 500)
        pdfs = map(lambda x:gamma_pdf(x, a,b), points)
        l, = plt.plot(points, pdfs, color=colors[i%len(colors)])
        '''
        pdfs = map(lambda x:gamma.pdf(x, a,b), points)
        l, = plt.plot(points, pdfs, color=colors[i%len(colors)])
        '''
        ls.append(l)
    

    plt.legend(handles = ls, labels = ab_lables, loc = 'best')
    plt.savefig(fname, format='png')

if __name__ == '__main__':
    ab_list = [(0.5, 0.5), (1, 0.5), (2, 0.5), (3, 0.5)]
    fname = 'images/gamma01.png'
    draw_gamma_dist(ab_list, fname, 50)
    
    plt.clf()
    ab_list = [(0.5, 1), (1, 1), (2, 1), (5, 1), (10, 1)]
    fname = 'images/gamma02.png'
    draw_gamma_dist(ab_list, fname, 20)

    plt.clf()
    ab_list = [(0.5, 1), (1, 1), (0.5, 2), (1, 2), (0.5,3), (1, 3)]
    fname = 'images/gamma03.png'
    draw_gamma_dist(ab_list, fname, 10)
    

    plt.clf()
    ab_list = [(9, 7), (7, 4)]
    fname = 'images/gamma04.png'
    draw_gamma_dist(ab_list, fname, 10)
