#encoding: utf8

import numpy as np
#注意下：chi和chi2是两个分布
from scipy.stats import chisquare,chi2,norm,poisson

import pdb

'''
分布检验

分布的卡方检验有推导，有空可以看下

对chi_square 检验而言，自由度为样本的区间个数 - 1(k-1个区间的概率定了后，剩下的区间概率就确定了)

fi: 实测第i个区间的频数
pi: 理论上该区间的概率. 如果没有，可以用估计值代替，这样自由度要减于估计参数的个数()
n*pi: 理论上该区间的频数. 当n*pi < 5时，要合并区间, 相应的自由度也要减1

xi = (fi - npi)^2/npi, 
X=sum(xi) ~ chi(x, df=k-1-b), b为估计参数的个数(减小误差)

p for 豌豆
'''

def chi_for_pea():
    pmf = np.array([9./16, 3./16, 3./16, 1./16])
    test_data = np.array([315, 101, 108, 32])
    test_sum = test_data.sum() * 1.

    df = len(pmf) - 1
    npi = test_sum * pmf
    X2 = ((test_data - npi) ** 2) * 1./npi
    chisq = X2.sum()
    p = 1 - chi2.cdf(chisq, df)

    print "Detail  : chisq for pea: %.4f; p value for pea: %.4f" % (chisq, p)

    #直接这么算也可以
    f_obs = test_data 
    f_exp = npi
    chisq, p = chisquare(f_obs, f_exp)
    print "Directed: chisq for pea: %.4f; p value for pea: %.4f" % (chisq, p)



'''
战争分布p值
X ~ Posisson(lambda)
x  0     1     2     3     4
fi 223   142   48    15    4
   0.58  0.31  0.18  0.01  0.02
n  216.7 149.5 51.6  12.0  2.16 

注意:
    1. posisson分布的参数是估计的，故而自由度在减1
    2. 第5年的npi < 5, 和前一个合并了，故而自由度又要减1

卡方分布时，如果对应的fi比较小(小于0.01)，则和上一个合并

x  0     1     2     3,4
fi 223   142   48    15+4
   0.58  0.31  0.18  0.01+0.02
n  216.7 149.5 51.6  12.0+2.16 
'''
def chi_for_war():
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
    chisq = stats[:-1].sum()
    
    #chi 计算
    df = 5 - 1 -1 - 1 # lambda_hat 是模拟数据，于是 df会减1; 合并最后一个 n*pi < 5, 于是又减了一个
    alpha_ppf = chi2.ppf(1-alpha, df=df)
    p = 1 - chi2.cdf(chisq, df=df)
    print "Detail  : chisq: %.4f; p: %.4f; alpha: %.4f; alpha_ppf: %.4f" % (chisq, p, alpha, alpha_ppf)

    #直接这么算也可以
    f_obs = fis[:-1]
    f_exp = npi[:-1]
    #lambda_hat是模拟的，在减去1
    chisq, p = chisquare(f_obs, f_exp, ddof=1)
    print "Directed: chisq: %.4f; p: %.4f; alpha: %.4f; alpha_ppf: %.4f" % (chisq, p, alpha, alpha_ppf)


if __name__ == "__main__":
    print "\np value for wars:"
    chi_for_war()
    
    print "\np value for peas:"
    chi_for_pea()
