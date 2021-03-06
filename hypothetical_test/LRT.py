#encoding: utf8

import numpy as np
from scipy.stats import chi2,norm,poisson

import pdb

'''
似然比检验, likelyhood ratio test
还是参数检验
wald test 适合标量检查;  LRT适合检测向量参数
lambda = 2 * log(sup(seta in all) L(seta) / sup(seta in SETA0) L(seta0))
       = 2 * log(L(seta_hat) / L(seta0_hat))

seta_hat == seta0_hat: lambda == 0
seta_hat >> seta0_hat: lambda >> 0

lambda ~ chi(r-q): r为seta的参数个数(自由度)， q为seta0参数个数

test for pea
'''

def lrt_for_pea():
    pmf = np.array([9./16, 3./16, 3./16, 1./16])
    test_data = np.array([315, 101, 108, 32])
    test_sum = test_data.sum() * 1.

    r = len(pmf) - 1
    q = 0

    seta_hat = test_data/test_sum 

    #sum(Xi * log(pi_hat / pi) * 2
    lambs = 2 * np.log(seta_hat/pmf) * test_data
    lamb = lambs.sum()
    
    p_val = 1 - chi2.cdf(lamb, r-q)

    print "p value for pea: %.4f" % p_val
    

'''
战争分布p值
X ~ Posisson(lambda)
x  0     1     2     3     4
fi 223   142   48    15    4
   0.58  0.31  0.18  0.01  0.02
n  216.7 149.5 51.6  12.0  2.16 

卡方分布时，如果对应的fi比较小(小于0.01)，则和上一个合并

x  0     1     2     3,4
fi 223   142   48    15+4
   0.58  0.31  0.18  0.01+0.02
n  216.7 149.5 51.6  12.0+2.16 
'''
def lrt_for_war():
    alpha = 0.05
    years = np.linspace(0, 4, 5)
    wars = np.array([223, 142, 48, 15, 4])
    wars_sum = wars.sum()


    #poission 参数估计, lambda_hat = 1/n * sum Xi
    lambda_hat = 1./wars_sum * np.dot(wars, years) 
    
    #频度计算
    fis = np.array(wars)
    fis[-2] += fis[-1]
    
    p_of_years = poisson.pmf(years, lambda_hat)
    npi = p_of_years * wars_sum
    npi[-2] += npi[-1]
    
    stats = np.power(fis - npi,  2) / npi
    stat = stats[:-1].sum()
    
    #chi 计算
    df = 5 - 1 -1 - 1 # lambda_hat 是模拟数据，于是 df会减1; 合并最后一个 n*pi < 5, 于是又减了一个
    #X2 = chi.ppf(1-alpha, df=df), 这个是错误的，因为chi分布不是对称的
    X2 = chi.ppf(1-alpha, df=df) ** 2


    print "stat: %.4f; X2: %.4f" % (stat, X2)


if __name__ == "__main__":
    lrt_for_pea()
