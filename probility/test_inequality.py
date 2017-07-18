#encoding: utf8

import pdb
import numpy as np

'''
测试不等式。
假定f(x) = x*x*3/1000, 0 <= x <= 10, 
这个分布中， X取值越大， pdf 越大.  从输出看，这个不等式预测效果是很差的

相比而言，这个还是对类似钟型分布、指数分布的曲线有效，对均匀分布、碗型分布之类的，比较差

'''

def call_probility_EV():
    # E = intgrate(x^3 * 3/1000), 3/1000/4 * 10**4
    E =  3./1000/4 * 10**4
    # E(X*X) = intgrate(x^4 * 3/1000), 3/1000/5 * 10**5
    V = 3./1000/5 * 10**5 - E*E

    return E,V

def call_probility(x):
    if x< 0:
        return 0, 0
    if x > 10:
        return 1, 0 

    CDF = 1/1000. * x**3
    pdf = 3./100 * x**2 

    return CDF,pdf

'''
P(X >= k) <= E/k

注意：这个要求X >= 0
'''
def test_markov(pf, evf):
    E, V = evf() 
    Elist = [E, 2*E, 3*E]
    Vlist = [V, 2*V, 3*V]

    Elist = [E, 8.5, 9, 9.5, 9.99]

    for x in Elist:
        #pdb.set_trace()
        CDF, pdf = call_probility(x)
        minus_CDF = 1. - CDF
        pk = E/x

        print "E, %s; x: %s; minus_CDF: %s; pk:%s; diff: %s" % (E, x, minus_CDF, pk, pk - minus_CDF)

'''
P(|X-mu| > k*sigma) <= 1/k**2
注意取mu两边的概率和

这个分布不好。导致计算时偏差很大
'''
def test_chebyshev(pf, evf):
    E, V = evf() 
    sigma = np.sqrt(V)
    Elist = [E, 2*E, 3*E]

    Vlist = [sigma, 2*sigma, 3*sigma]

    for x in Vlist: 
        xg = E + x
        xl = E - x
        #pdb.set_trace()
        #本分布函数不包括负数

        CDF1, pdf1 = call_probility(xg)
        CDF2, pdf2 = call_probility(xl)
        minus_CDF = 1. - CDF1 + CDF2  # X > sg, X < xl的概率和
        k = x/sigma
        pk = 1.0/(k*k)

        print "V, %s; sigma: %s, x: %s; minus_CDF: %s; pk:%s; diff: %s" % (V, sigma, x, minus_CDF, pk, pk - minus_CDF)

if __name__ == "__main__":
    #pdb.set_trace()
    for x in range(0, 11):
        CDF,pdf = call_probility(x)
        E, V = call_probility_EV() 
        print "x:", x, CDF, pdf, E,V

    print "\n\nDO TEST:"
    print "Markov TEST:"
    test_markov(call_probility, call_probility_EV)

    print "chebyshev TEST:"
    test_chebyshev(call_probility, call_probility_EV)
