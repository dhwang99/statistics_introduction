#encoding: utf8

import numpy as np
import pdb

'''
用poisson分布模拟二项分布。条件：
  1. p < 0.1 或 (1-p) < 0.1
  2. np < 5

  二项分布: C(n, m)p^m * (1-p)^(n-m)
'''

'''
return CDF, pmf
'''
def get_poisson(lmb, m):
    CDF = 0.0
    pdf = 0.0
    k_f = 1  # factorial to k
    for k in xrange(0, m+1):
        if k > 0:
            k_f *= k   #k!

        pdf = np.power(lmb, k)*1.0/k_f * np.exp(-lmb)
        CDF += pdf
    return pdf, CDF 

def get_Binomial(p, m, n):
    CDF = 0.0
    pdf = 0.0
    k_f = 1  # factorial to k
    n_f = 1  # factorial from n to n-k+1
    for k in xrange(0, m+1):
        if k > 0:
            k_f *= k   #k!
            n_f *= (n - k + 1)

        pdf = 1. * n_f/k_f * np.power(p, k) * np.power((1. - p), n - k)
        CDF += pdf

    return pdf, CDF

if __name__ == "__main__":
    p_list = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95] 
    n_list = [10, 20, 40, 60, 80, 90]

    p = 0.1
    lmb = 1
    p_p1 = get_poisson(lmb, 1)
    p_p5 = get_poisson(lmb, 5)
    p_p10 = get_poisson(lmb, 10)
    p_b1 = get_Binomial(p, 1, 10)
    p_b5 = get_Binomial(p, 5, 10)
    p_b10 = get_Binomial(p, 10, 10)

    print "poisson:", p_p1, p_p5, p_p10
    print "Binomial:", p_b1, p_b5, p_b10
    
    for p in p_list:
        for n in n_list:
            lmb = n * p
            rst_list = []
            for m in range(0, 10):
                CDF_p, pdf_p = get_poisson(lmb, m)
                CDF_b, pdf_b = get_Binomial(p, m, n)

                rst_list.append((p, n, m, CDF_p, pdf_p, CDF_b, pdf_b, CDF_p-CDF_b, pdf_p-pdf_b))
            
            format_str = 'p=%s n=%s m=%s : ' + ' '.join(['%.4f']*6)
            rst_str = '  \n'.join(map(lambda x:format_str%x, rst_list))
            print rst_str
            print ''

