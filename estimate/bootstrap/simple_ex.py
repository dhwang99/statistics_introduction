#encoding: utf8

import numpy as np

'''
lim(1-1/x)^x = 1/e
'''
def sample_missing_probility(N):
    hit_pos = np.zeros(N)
    samples = np.random.randint(low=1, high=N, size=100)
    
    for n in samples:
        hit_pos[n] += 1
    
    no_hit = 0
    for i in hit_pos:
        if i == 0:
            no_hit += 1
    
    E_no_hit = (1 - 1./N)**N
    no_hit /= (1. * N)
    print "N: %s; No Hit: %s; expect: %.3f; real ratio: %.3f" % (N, no_hit, E_no_hit, no_hit)
    print "no_hit minus e:", (no_hit - 1/np.e)

    return no_hit
    
'''
1. 采样：X1 ... Xn ~ hat_Fn
2. 计算 Tn ~ g(X1, ..., Xn)
3. 重复 step1,2 B次， 得到T1, ..., Tnstar的B个样本
4. 求B的均值和方差

从结果看，N太小时，估计的效果不好
'''
def estate_mean_Var():
    samples = np.array([3.12, 0, 1.57, 19.67, 0.22, 2.20])
    bar_X = samples.mean()

    B_list = [5, 10, 50]
    N = samples.shape[0]
    
    print "X_mean:", bar_X
    for B in B_list:
        X_star_means = np.zeros(B)
        for bid in xrange(B):
            X_star = np.zeros(N)
            for i in xrange(N):
                rid = np.random.randint(0, N)
                X_star[i] = samples[rid]
            X_star_means[bid] = X_star.mean()

        print "B: %s; X_star_mean: %s; var:%s" % (B, X_star_means.mean(), X_star_means.var())
        
if __name__ == "__main__":
    N = 100
    sm = 10
    rst = np.zeros(sm)

    for i in xrange(sm):
        rst[i] = sample_missing_probility(N)

    print "Result. mean: %s; std error: %s" % (rst.mean(), rst.std()) 

    estate_mean_Var()
