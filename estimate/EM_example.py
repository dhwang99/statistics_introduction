#encoding: utf8

'''
EM求解混合高斯模型
例子描述如下：统计得到n个学生的身高,且男女身高符合正态分布
求：男女身高分布参数，男女比例参数(p)

留下的问题：
1. 方差参数估计
2. 随机取样方法
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
'''
def pdf_for_mix_norm(xi, param):
    p1, mu1, sigma1, mu2, sigma2 = param
    pdf1 = pdf_for_norm(xi, mu1, sigma1)
    pdf2 = pdf_for_norm(xi, mu2, sigma2)

    pdf = p1 * pdf1 + (1. - p1) * pdf2 

    return pdf

'''
bounds like (1.5, 1.8)
'''
def plot_mix_norm_distribute(param, bounds, color = 'k'):
    bins = np.linspace(bounds[0], bounds[1], 100) 
    pdfs = map(lambda x:pdf_for_mix_norm(x, param), bins)
    plt.plot(bins, pdfs, color=color)


'''
p为男学生的概率，采样为二项式分布(逐个(0, 1)分布)
'''
def gen_samples(n, param):
    #n1, 为男学生的采样数
    p1, mu1, sigma1, mu2, sigma2 = param
    n1 = np.random.binomial(n, p1, 1)[0]
   
    #生成男学生的样本, 符合mu1, sigma1^2 的高斯分布
    sample1 = np.random.normal(mu1, sigma1, n1)
    #生成女学生的样本
    sample2 = np.random.normal(mu2, sigma2, n-n1)

    return np.hstack((sample1, sample2))



'''
简化版的算法
重要的概念是, 对xi, 求其属于k的比例
p(xi,zk) = p(xi|zk)*p(zk) = alpha_k * phi(x_i|mu_k, sigma_k^2)
L = sum_n sum_k p(xi, zk) = sum_n sum_k alpha_k * phi(x_i|mu_k, sigma_k^2)

gamma(i,k) = p(zk|xi) = p(xi, zk)/p(xi)
'''

def EM_1(samples):
    alpha = 0.5
    mu1 = 1.7
    sigma1 = 0.2  #标准差小点
    mu2 = 1.5
    sigma2 = 0.2  #小短腿、大长腿比例要高一些
    k = 2          #隐变量分类数
   
    N = len(samples)
    loop_count = 100
    l = 0
    while l < loop_count:
        l+=1
        #E step
        N_1 = N_2 = 0.

        x_k1 = np.zeros(N)
        x_k2 = np.zeros(N)

        gamma_k1 = np.zeros(N)
        gamma_k2 = np.zeros(N)
        for i in xrange(N):
            xi = samples[i]
            # calcuate p(xi, zk), Z,X独立
            p_xz_i1 = alpha * pdf_for_norm(xi, mu1, sigma1)
            p_xz_i2 = (1.-alpha) * pdf_for_norm(xi, mu2, sigma2)
            #p(xi) = p(xi,z1) + p(xi, z2)
            p_xi = p_xz_i1 + p_xz_i2 
            # p(zk|xi)
            p_z1_xi = p_xz_i1 / p_xi
            p_z2_xi = p_xz_i2 / p_xi 

            N_1 += p_z1_xi
            N_2 += p_z2_xi

            x_k1[i] = p_z1_xi * xi
            x_k2[i] = p_z2_xi * xi

            gamma_k1[i] = p_z1_xi
            gamma_k2[i] = p_z2_xi

        mu_es1 = x_k1.sum() / N_1   #注意不是 除以N, 是除以N_1, 下同
        mu_es2 = x_k2.sum() / N_2

        # sigma_es1 = x_k1.var(), sigma_es2 = x_k2.var(), 这个是错的。同上
        # 求sigma 这一块有疑问
        sigma_es1 = np.dot(gamma_k1 * (samples - mu_es1), (samples - mu_es1).T) / N_1
        sigma_es2 = np.dot(gamma_k2 * (samples - mu_es2), (samples - mu_es2).T) / N_2
        alpha_es = N_1 / N
        #pdb.set_trace()

        mu1 = mu_es1
        mu2 = mu_es2
        sigma1 = np.sqrt(sigma_es1)
        sigma2 = np.sqrt(sigma_es2)
        alpha = alpha_es

    return alpha, mu1, sigma1, mu2, sigma2

if __name__ == "__main__":
    init_param = (0.4, 1.7, 0.2, 1.5, 0.25)    #这一组非常不好区分。两个分布的重合度非常高，导致算不准
    init_param = (0.4, 1.75, 0.1, 1.5, 0.15)
    init_param = [0.4, 1.85, 0.1, 1.4, 0.15]
    n = 2000
    samples = gen_samples(n, init_param) 

    rst = EM_1(samples)

    print "init param:", init_param
    print rst

    bounds = (1.1, 2.1)
    plt.clf()
    plot_mix_norm_distribute(init_param, bounds, color='r')
    plot_mix_norm_distribute(rst, bounds, color='b')

    test_p1 = list(init_param)
    test_p1[0] = 0
    plot_mix_norm_distribute(test_p1, bounds, color='y')
    test_p1[0] = 1
    plot_mix_norm_distribute(test_p1, bounds, color='c')
    plt.savefig('images/mix_norm/distibute.png', format='png')
