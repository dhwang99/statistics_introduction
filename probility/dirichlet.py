#encoding: utf8

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D

from gamma_dist import gamma_values
import pdb

'''
Dir(X;alpha_vector) = (Gamma(sum(alpha_vector))/Multi(Gamma(alpha_i)) * Multi(xi^alpha_i)
'''
def dir_pdf(alpha_vector, X_vector): 
    #pdb.set_trace()
    s = X_vector.sum()

    return stats.dirichlet.pdf(X_vector, alpha_vector)

    alpha_sum = alpha_vector.sum()
    gamma_sum = gamma_values[alpha_sum]
    gamma_multi = reduce (lambda x,y:x*y, map(lambda x:gamma_values[x], alpha_vector))
    px_multi = np.prod(np.power(X_vector, alpha_vector-1))

    pdf = gamma_sum / gamma_multi * px_multi 
    if pdf < 0:
        pdf = 0

    return pdf

def dir_pdfs(alpha_vector, X, Y):
    m,n = X.shape
    pdfs = np.zeros((m,n))
    for i in xrange(m):
        for j in xrange(n):
            z = 1 - X[i,j] - Y[i,j]
            if z > 0:
                pdfs[i,j] = dir_pdf(alpha_vector, np.array([X[i,j], Y[i,j], z])) 

    return pdfs



def get_3d_points():
    X = np.linspace(0.01, .99, 100)
    Y = np.linspace(0.01, 0.99, 100)

    return np.meshgrid(X, Y)

'''
only for 3-d
'''
def draw_dir_dist(alpha_vector, fname):
    colors = ['r', 'b', 'k', 'g', 'm', 'c']
    ls = []
    ab_lables = '%s:%s:%s' % (alpha_vector[0], alpha_vector[1], alpha_vector[2])

    a_v = alpha_vector
    X,Y = get_3d_points()  # sum(xi)=1
    '''
    pdfs = map(lambda x:beta_pdf(x, a,b), points)
    l, = plt.plot(points, pdfs, color=colors[i%len(colors)])
    '''

    pdfs = dir_pdfs(a_v, X, Y)


    '''
    R = np.sqrt(X**2 + Y**2)
    pdfs = np.sin(R)
    '''

    #pdb.set_trace()

    fig = plt.figure()
    ax = Axes3D(fig)

    #画三维图
    #ax.plot_surface(X, Y, pdfs, rstride=1, cstride=1, cmap='rainbow')
    surf = ax.plot_surface(X, Y, pdfs, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False)
    
    plt.savefig(fname, format='png')


if __name__ == "__main__":
    a_v = np.array([0.1, 0.1, 0.1])
    fname = "images/dir0.png"
    draw_dir_dist(a_v, fname)

    a_v = np.array([0.5, 0.5, 0.5])
    fname = "images/dir1.png"
    draw_dir_dist(a_v, fname)

    a_v = np.array([1, 1, 1])
    fname = "images/dir2.png"
    draw_dir_dist(a_v, fname)

    a_v = np.array([5, 5, 10])
    fname = "images/dir4.png"
    draw_dir_dist(a_v, fname)

    a_v = np.array([10, 10, 10])
    fname = "images/dir3.png"
    draw_dir_dist(a_v, fname)
