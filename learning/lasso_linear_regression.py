#encoding: utf8

import numpy as np
import pdb
import matplotlib.pyplot as plt

from data_loader import load_data

from cv_common import one_std_error_rule, gen_CV_samples_by_K_folds 


'''
在特征之间有一定线性相关度时（协方差不为0），一个大的特征参数有可能会增加而另一些变为负数或趋于0
这导致方差很大
L1通过收缩参数，减小了相关特征的影响，部分特征的参数项可以为0
Y = beta * X + lambda * abs(beta)

 用k折交叉验证法进行df选择
 注意下：
 1. 如果不用bias/intercept, 对Y要进行均值化
 2. numpy 的 svd, 返回的是V的转秩，用的时候需要再转秩一下
 3. 求估计(均值)的标准差时，记得除以sqrt(n)
 4. 对数据标准化时，倾向于用全局数据，并除以方差，即: xi = (xi - mean_hat)/sigma_hat
    不过本例子给的示例只做了中心标准化，未除以方差（个人觉得需要除以方差）
    但在全局子集选择里，CV方法下又做了标准化（即除了方差）
    稍后这儿也试一下标准化后的结果
'''
def lasso_leasq():
    '''
    注：实际训练时，标准化应该针对X_train_CV进行，不能把X_test_CV加进来, 避免影响样本的独立性
        本试验为了方便，把这个加起来了
    '''
    X_train, Y_train, X_test, Y_test, dt_stand_fun = load_data(type=1, need_bias=0, y_standard=1)

    K = 10
    lamb_lst = np.logspace(-5, 3, 30)
    cv_mse_mean_hat = np.zeros(len(lamb_lst))
    cv_mse_hat_se = np.zeros(len(lamb_lst))

    cv_samples = gen_CV_samples_by_K_folds(X_train, Y_train, K)

    for lid in lamb_lst:
        lamb = lamb_lst[lid]
        test_mses = np.zeros(K)
        for ki in range(K):
            ki_rst = []
            X_CV, Y_CV, X_t_CV, Y_t_CV = cv_samples[ki]
            # wait for coding
    
            ki_rst.append((lamb, beta_hat, df, train_rss, test_rss))
        
        train_mid_rst.append(ki_rst)

    #计算不同lamb下的训练误差和方差
    dfs = np.zeros(len(lamb_lst))
    rss_means = np.zeros(len(lamb_lst))
    rss_mean_stds = np.zeros(len(lamb_lst))

    for lid in range(len(lamb_lst)):
        #lambda[lid]下的df
        dfs[lid] = train_mid_rst[0][lid][2]
        #K折CV下误差均值和标准差
        test_rsses = np.array(map(lambda i:train_mid_rst[i][lid][4], range(0,K)))
        rss_means[lid] = test_rsses.mean()
        #!!!!!注意：这儿求的是估计的标准差 1/K mean(sum(Xi)),  故而要除以K
        rss_mean_stds[lid] = test_rsses.std()/np.sqrt(K)

    best_lamb_id, minid = one_std_error_rule(rss_means, rss_mean_stds)
    print "Best lambid: %d, lambda: %.4f, degree of free: %.4f" % (best_lamb_id, lamb_lst[best_lamb_id], dfs[best_lamb_id]) 
    
    one_std_val = rss_means[minid] + rss_mean_stds[minid]
    plt.plot((dfs[0],dfs[-1]), (one_std_val, one_std_val), 'r-')
    plt.errorbar(dfs, rss_means, yerr=rss_mean_stds, fmt='-o')
    plt.savefig('images/rss_errorbar.png', format='png')

    #用K折选出来的最优lambda进行回归预测

if __name__ == '__main__':
    print "ridge leasq:"
    lasso_leasq()
    print ""
