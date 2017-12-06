#encoding: utf8

import pdb
import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_data

from cv_common import one_std_error_rule, gen_CV_samples_by_K_folds 


'''
在特征之间有一定线性相关度时（协方差不为0），一个大的特征参数有可能会增加而另一些变为负数或趋于0
这导致方差很大
L1通过收缩参数，减小了相关特征的影响，部分特征的参数项可以为0

f(x) = seta * x

J(X;seta) = 1/2n*(f(X) - Y)**2  + lambda * np.linalg.norm(seta, 1)

1. 坐标下降法求解
    psj = partial J/partial seta_j 
        = 1/n*sum_i[(f(X_i) - Y_i)*X_ij] + r_l1
        = 1/n*sum_i[(seta * X_i - Y_i) * X_ij] + r_l1
        = 1/n*sum_i[sum_k(seta_k * X_ik<k!=j>)*X_ij + seta_j * X_ij**2 - Y_i*X_ij] + r_l1 
        or 
        = 1/n*sum_i[(seta * X_i - Y_i) * X_ij - seta_j * X_ij**2 + seta_j*X_ij**2] + r_l1

    let: 
        p_j = 1/n*sum_i[(Y_i - seta * X_i) * X_ij + seta_j * X_ij**2]
        z_j = 1/n*sum_i(X_ij**2)

    ps = -p_j + seta_j*z_j + r_l1

    r_l1 = lambda, seta_j > 0
           [-lambda, lambda], seta_j = 0
           -lambda, seta_j < 0

    seta_j = (p_j - lambd)/z_j, if p_j > lambd
           = (p_j + lambd)/z_j, if p_j < -lambd
           = 0, else


2. 最小角回归求解(LARS)
    waiting

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

def lasso_cd(X, Y, lamb):
    it_count = 5000
    epilson = 1e-6

    n,m = X.shape
    seta = np.ones(m)

    mse = 1e10
    mse_new = 1e10

    for it_i in xrange(it_count):
        for j in xrange(m):
            Xj2Xj = np.dot(X[:,j], X[:,j])
            p_j = 1./n * (np.dot(Y-np.dot(X, seta), X[:,j]) + seta[j]*Xj2Xj)
            z_j = 1./n * Xj2Xj 

            if p_j > lamb:
                seta[j] = (p_j - lamb)/z_j
            elif p_j < -lamb:
                seta[j] = (p_j + lamb)/z_j
            else:
                seta[j] = 0.

        err1 = np.dot(X, seta) - Y 
        mse_new = np.dot(err1, err1) / n
        if np.abs(mse_new - mse) < epilson:
            break
        mse = mse_new

    return seta, mse_new



def lasso_leasq_cd_CV():
    '''
    注：实际训练时，标准化应该针对train_X_CV进行，不能把test_X_CV加进来, 避免影响样本的独立性
        本试验为了方便，把这个加起来了
    '''
    train_X, train_Y, test_X, test_Y, dt_stand_fun = load_data(type=1, need_bias=0, y_standard=1)

    K = 10
    lamb_lst = np.logspace(-3, 0, 100)
    train_mid_rst = []
    cv_samples = gen_CV_samples_by_K_folds(train_X, train_Y, K)

    for lid in xrange(len(lamb_lst)):
        lamb = lamb_lst[lid]
        test_mses = np.zeros(K)
        ki_rst = []
        for ki in range(K):
            X_CV, Y_CV, X_t_CV, Y_t_CV = cv_samples[ki]
            # wait for coding
            seta, train_mse = lasso_cd(X_CV, Y_CV, lamb) 
            y_hat_err = np.dot(X_t_CV, seta) - Y_t_CV
            test_mse = np.dot(y_hat_err, y_hat_err) / len(Y_t_CV) 

            df = len(np.where(np.abs(seta) < 1e-5)[0])
    
            ki_rst.append((lamb, seta, df, train_mse, test_mse))
        
        train_mid_rst.append(ki_rst)

    #计算不同lamb下的训练误差和方差
    dfs = np.zeros(len(lamb_lst))
    mse_means = np.zeros(len(lamb_lst))
    mse_mean_stds = np.zeros(len(lamb_lst))

    for lid in range(len(lamb_lst)):
        #K折CV下误差均值和标准差
        test_msees = np.array(map(lambda i:train_mid_rst[lid][i][4], range(0,K)))
        train_msees = np.array(map(lambda i:train_mid_rst[lid][i][3], range(0,K)))
        mse_means[lid] = test_msees.mean()
        #!!!!!注意：这儿求的是估计的标准差 1/K mean(sum(Xi)),  故而要除以K
        mse_mean_stds[lid] = test_msees.std()/np.sqrt(K)

        print "lasso CD for lambda: %.4f, CV train mse: %.4f, test mse: %.4f, std: %.4f" % \
                (lamb_lst[lid], train_msees.mean(), mse_means[lid], mse_mean_stds[lid])

    '''
    #一倍方差准则
    '''
    best_lamb_id, minid = one_std_error_rule(mse_means, mse_mean_stds)
    best_lamb = lamb_lst[best_lamb_id]
    print "Best lambid: %d, lambda: %.4f, degree of free: %.4f" % (best_lamb_id, best_lamb, dfs[best_lamb_id]) 
    
    one_std_val = mse_means[minid] + mse_mean_stds[minid]
    plt.plot((dfs[0],dfs[-1]), (one_std_val, one_std_val), 'r-')
    plt.errorbar(dfs, mse_means, yerr=mse_mean_stds, fmt='-o')
    plt.savefig('images/lasso_mse_errorbar.png', format='png')

    #用K折选出来的最优 lambda 进行回归预测
    '''
    #非一倍方差准则
    best_lamb_id = np.argmin(mse_means)
    best_lamb = lamb_lst[best_lamb_id]
    '''

    seta, train_mse = lasso_cd(train_X, train_Y, best_lamb)

    y_hat_err = np.dot(test_X, seta) - test_Y 
    test_mse = np.dot(y_hat_err, y_hat_err) / len(test_Y) 

    print "Test error: train mse: %.4f, test mse: %.4f" % (train_mse, test_mse)

    print "seta: %s" % seta



if __name__ == '__main__':
    print "lasso leasq by corr descent:"
    lasso_leasq_cd_CV()
    print ""
