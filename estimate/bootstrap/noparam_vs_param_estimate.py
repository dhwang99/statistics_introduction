#encoding: utf8

import numpy as np
import matplotlib.pyplot as plt

import pdb

def uniform_by_nonparam(samples, B):
    X_star_samples = np.zeros(B)
    n = len(samples)
    bn = np.random.randint(0, high=n, size=(n, B))
    #X_star_samples = np.max(np.take(samples, bn), axis=0)
    for b in xrange(B):
        X_star_samples[b] = np.max(np.take(samples, bn[:,b]))

    group_count = int(np.sqrt(B))
    plt.hist(X_star_samples, bins = group_count, normed=True, color='r', alpha=0.6)

    print "X_star_samples mean,std:", X_star_samples.mean(), X_star_samples.std()

    return

'''
1. 先估计参数 得到 hat_seta
2. 用 hat_seta采样n个（如10）
3. 对 100个样本，用估计函数进行估计，得到估计值 Ti,n
4. 重复2,3 B次，得到B个Tn样本
5. 计算估计的方差
'''
def uniform_by_param(samples, B):
    X_star_samples = np.zeros(B)
    n = 100  #这个合理不？
    n = len(samples)   # n=100会明显降低估计的标准差，不合理
    theta = np.max(samples)
     
    Tn_samples = np.random.random((n, B)) * theta
    X_star_samples = np.max(Tn_samples, axis=0)

    group_count = int(np.sqrt(B))
    plt.hist(X_star_samples, bins = group_count, normed=True, color='k', alpha=0.6)
    print "X_star_samples mean, std:", X_star_samples.mean(), X_star_samples.std()
    print "Theta:", theta


if __name__ == "__main__":
    samples = np.random.random(20)
    B = 1000

    print "uniform_by_nonparam:"
    uniform_by_nonparam(samples, B)
    
    print ""
    print "uniform_by_param:"
    uniform_by_param(samples, B)

    plt.savefig('images/uniform_vs_non_param.png', format='png')

