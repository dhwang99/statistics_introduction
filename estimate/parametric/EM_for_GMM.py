#encoding: utf8

'''
EM求解混合高斯模型
例子描述如下：统计得到n个学生的身高,且男女身高符合正态分布
求：男女身高分布参数，男女比例参数(p)

留下的问题：
1. 初始参数预估。现在给的是一个人工猜的方法
2. 随机取样方法。
3. 多元正态分布的计算, 样本都是一元正态的（学生身高)
这些回头补一补
'''

import numpy as np
import pdb
import matplotlib.pyplot as plt

'''
正态分布的pdf
'''
def pdf_for_norm(xi, mu, sigma):
    pdf = 1./(np.sqrt(2*np.pi)*sigma) * np.exp(-(xi-mu)**2/(2*sigma**2)) 
    return pdf

'''
混合正态分布的pdf
param: list for elements: (pi, mu_i, sigma_i), sum(pi) = 1 
'''
def pdf_for_mix_norm(xi, params):
    pdf = 0.
    for param_i in params:
        p_i, mu_i, sigma_i = param_i
        pdf_i = pdf_for_norm(xi, mu_i, sigma_i)
        pdf += p_i * pdf_i

    return pdf

'''
bounds like (0.0, 3.0)
'''
def plot_mix_norm_distribute(param, bounds, color = 'k'):
    bins = np.linspace(bounds[0], bounds[1], 100) 
    pdfs = map(lambda x:pdf_for_mix_norm(x, param), bins)
    plt.plot(bins, pdfs, color=color)


'''
如果只有男女学生，采样为二项式分布(逐个(0, 1)分布)
为了兼容多个分组，本示例用的是多项式分布
'''
def gen_samples(n, params):
    sample_counts = np.random.multinomial(n, params[:, 0], 1)[0]
    samples = []
    i = 0
    for i in xrange(len(params)):
        p_i, mu_i, sigma_i = params[i]
        #生成学生的样本, 符合mu1, sigma1 的高斯分布
        sample_i = np.random.normal(mu_i, sigma_i, sample_counts[i])
        samples.append(sample_i)

    return np.hstack(samples)


'''
EM求解 Gaussian Mixture Model

p(xi,zk) = p(xi|zk)*p(zk) = phi(x_i|mu_k, sigma_k^2) * psi_k

w(i,k) = p(zk|xi) = p(xi, zk)/p(xi)

l = sum_n sum_k p(xi, zk) = sum_n(sum_k(w(zk,xi) * log(p(xi, zk)/w(zk,xi))))
'''
def EM_for_GMM(samples, init_params, loop_count): 
    m = len(samples)
    k = len(init_params)
    params = np.copy(init_params)
    x = samples

    for l in xrange(loop_count):
        w_a = np.zeros((m, k))
        #E step
        for i in xrange(m):
            xi = x[i]
            p_xi = pdf_for_mix_norm(xi, params) 
            for j in xrange(k):
                psi_j, mu_j, sigma_j = params[j]
                p_xi_given_yk = pdf_for_norm(xi, mu_j, sigma_j)  #N(xi;mu_j, sigma_j) 
                p_xi_yj = p_xi_given_yk * psi_j   # p(xi,yj) = p(xi|yj)*p(yj) 
                w_a[i,j] = p_xi_yj / p_xi         # p(yj|xi) = p(xi,yj)/p(xi) 

        #M step, 可考虑改为矩阵计算方法
        for j in xrange(k):
            Nj = np.sum(w_a[:, j], 0)
            mu_j = np.sum(w_a[:, j] * samples) / Nj
            sigma2_j = np.sum(np.dot((w_a[:,j] * (x - mu_j)), (x - mu_j))) / Nj

            params[j,:] = np.array([Nj/m, mu_j, np.sqrt(sigma2_j)])

    return params

def test_mix_norm_model(m, loop_count):
    sample_params = np.array([[0.4, 1.7,  0.2], [0.6, 1.5, 0.25]])    #这一组非常不好区分。两个分布的重合度非常高，导致算不准
    sample_params = np.array([[0.4, 1.75, 0.1], [0.6, 1.5, 0.15]])    
    sample_params = np.array([[0.4, 1.85, 0.1], [0.6, 1.4, 0.15]])   #这一组比较好处理

    samples = gen_samples(m, sample_params) 
    
    #生成初始参数
    rd = np.random.rand(sample_params.shape[0], sample_params.shape[1]) - 0.5
    init_params = sample_params * (1 - rd * 0.1)
    init_params[:,0] = 1./len(init_params)  #默认各分类下的概率相等

    sim_rst = EM_for_GMM(samples, init_params, loop_count)

    print "for  Sample count:", m
    print "Sample param:\n", sample_params
    print "init param:\n", init_params
    print "EM result:\n", sim_rst
    print ""

    bounds = (0.5, 2.5)
    plt.clf()
    plot_mix_norm_distribute(sample_params, bounds, color='r')
    plot_mix_norm_distribute(init_params, bounds, color='b')
    plot_mix_norm_distribute(sim_rst, bounds, color='k')

    '''
    test_p1 = init_params
    test_p1[:, 0] = [0,1]
    plot_mix_norm_distribute(test_p1, bounds, color='y')
    test_p1[:, 0] = [1,0]
    plot_mix_norm_distribute(test_p1, bounds, color='c')
    '''
    plt.savefig('images/mix_norm/mix_norm_distibute_%s.png' % m, format='png')

if __name__ == "__main__":
    mlist = [ 200,500, 1000, 2000, 5000]
    loop_count = 200
    for m in mlist:
        test_mix_norm_model(m, loop_count)
