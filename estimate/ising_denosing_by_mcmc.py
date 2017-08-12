#encoding: utf8

from PIL import Image
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pdb

'''
add noising to img
'''
def gen_noising_img(sigma = 1., filename='lettera.bmp'):
    img = Image.open(filename)    
    img_arr = np.array(img)
    if img_arr.ndim == 3:
        img_arr = img_arr[:,:,0]

    img_arr = img_arr + 0. 
    M,N = img_arr.shape
    img_arr = 2 * ((img_arr - img_arr.mean()) > 0) - 1.  # to 1. and -1.
    img_src = np.copy(img_arr)

    #generate noising data
    img_arr += norm.rvs(size=img_arr.shape, loc=0, scale=sigma)
    local_evidence = np.ones((M, N, 2))
    local_evidence[:,:, 0] = norm.pdf(img_arr, loc = -1, scale=sigma)
    local_evidence[:,:, 1] = norm.pdf(img_arr, loc = 1, scale=sigma)
    
    #generate init data
    init_X = 2. * (local_evidence[:,:, 0] < local_evidence[:, :, 1]) - 1
    
    return img_arr, init_X, local_evidence


def denosing_by_metropolis(y, init_X, J, N, sigma):
    X = np.copy(init_X)
    X0_size = X.shape[0]
    X1_size = X.shape[1]

    for i in xrange(N):
        x0_i = np.random.randint(X0_size)
        x1_i = np.random.randint(X1_size)

        pos = x0_i * X0_size + x1_i
        neighborhood = pos + np.array([-1,1,-X0_size,X0_size])
        avail_pos = np.where([x1_i>0, x1_i<X1_size-1, x0_i > 0, x0_i<X0_size-1])
        neighborhood = np.take(neighborhood, avail_pos)

        cur_state = X[x0_i, x1_i]
        #pdb.set_trace()
        e_now = np.sum(np.take(X, neighborhood) == cur_state)
        e_can = np.sum(np.take(X, neighborhood) != cur_state)

        delLogPr = np.exp(2 * J * (e_can - e_now))
        likeRat = np.exp((-y[x0_i, x1_i] * 2. * X[x0_i, x1_i])/(sigma**2))
        alpha = delLogPr * likeRat

        if np.random.random() < alpha:
            X[x0_i, x1_i] = -cur_state

    return X


def denosing_by_gibbs(y, init_X, le, J, N, sigma):
    X = np.copy(init_X)
    X0_size = X.shape[0]
    X1_size = X.shape[1]

    for i in xrange(N):
        x0_i = np.random.randint(X0_size)
        x1_i = np.random.randint(X1_size)

        pos = x0_i * X0_size + x1_i
        neighborhood = pos + np.array([-1,1,-X0_size,X0_size])
        avail_pos = np.where([x1_i>0, x1_i<X1_size-1, x0_i > 0, x0_i<X0_size-1])
        neighborhood = np.take(neighborhood, avail_pos)

        cur_state = X[x0_i, x1_i]
        #pdb.set_trace()
        wi = np.sum(np.take(X, neighborhood))

        p_1 = np.exp(2*J*wi) * le[x0_i, x1_i, 1]   #p(theta|y) 
        p_n1 = np.exp(-2*J*wi) * le[x0_i, x1_i, 0]

        p1 = p_1 / (p_1 + p_n1 + 1e-19)

        if np.random.random() < p1:
            X[x0_i, x1_i] = 1
        else:
            X[x0_i, x1_i] = -1 

    return X


if __name__ == "__main__":
    y, init_X, le  = gen_noising_img()
    plt.imshow(init_X, cmap='gray')
    plt.savefig('images/la_noised.png', format='png')

    J = 5
    N = 200000
    sigma = 1.
    X = denosing_by_metropolis(y, init_X, J, N, sigma) 

    plt.clf()
    plt.imshow(X, cmap='gray')
    plt.savefig('images/la_denoising_by_metropolis.png', format='png')

    X = denosing_by_gibbs(y, init_X, le, J, N, sigma) 
    plt.clf()
    plt.imshow(X, cmap='gray')
    plt.savefig('images/la_denoising_by_gibbs.png', format='png')


