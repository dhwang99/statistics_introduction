#encoding: utf8

import numpy as np
import pdb
import matplotlib.pyplot as plt

from data_loader import load_data

from cv_common import one_std_error_rule, gen_CV_samples_by_K_folds 


'''
在特征之间有一定线性相关度时（协方差不为0），一个大的特征参数有可能会增加而另一些变为负数或趋于0
这导致方差很大
L2通过收缩参数，减小了方差的影响
Y = beta * X + lambda * sum(beta^2)
 beta_hat = inv(X.T*X -lamb*I)*X.T*Y
用svd分解求解
 X = UDV.T
 beta_hat = V*inv(D*D + lamb*I)*D*U.T * y
 df(lamb) = sum [dj**2/(dj**2 + lamb)]

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
def leasq_with_L2_new():
    '''
    注：实际训练时，标准化应该针对X_train_CV进行，不能把X_test_CV加进来, 避免影响样本的独立性
        本试验为了方便，把这个加起来了
    '''
    X_train, Y_train, X_test, Y_test, dt_stand_fun = load_data(type=1, need_bias=0, y_standard=1)

    K = 10
    lamb_lst = np.logspace(-5, 3, 30)
    train_mid_rst = []
    cv_samples = gen_CV_samples_by_K_folds(X_train, Y_train, K)

    for ki in range(K):
        ki_rst = []
        X, Y, X_test_CV, Y_test_CV = cv_samples[ki]
        U,D,V = np.linalg.svd(X)
        V = V.T
        D2 = D**2
        D_UT_y = np.dot(np.dot(np.diag(D), U.T[:len(D2),:]), Y)
        I_b = np.eye(X.shape[1])

        for lamb in lamb_lst:
            #计算beta估计
            '''
            开始时算得有问题，原因:
            svd分解时，返回的是VT, 不是V, 用的时候需要注意一下.fix这个bug后，beta_hat2, beta_hat4, beta_hat基本一致了
            beta_hat2 = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + I_b * lamb), X.T), Y)
            beta_hat4 = np.dot(np.dot(V, np.linalg.inv(np.diag(D2) + I_b * lamb)), D_UT_y)
            beta_hat5 = np.dot(np.dot(V, np.diag(np.diag(1/(np.diag(D2) + I_b * lamb)))), D_UT_y)
            '''
            beta_hat = np.dot(np.dot(V, np.diag(1/(D2 + lamb))), D_UT_y)
    
            #计算训练误差
            train_err = np.dot(X, beta_hat.T) - Y
            train_rss = np.dot(train_err, train_err)/len(Y)
    
            #计算预测误差 
            test_err = np.dot(X_test_CV, beta_hat) - Y_test_CV
            test_rss = np.dot(test_err, test_err) /len(Y_test_CV)
    
            #计算自由度
            df = np.sum(D2/(D2+lamb))
    
            ki_rst.append((lamb, beta_hat, df, train_rss, test_rss))
        
        train_mid_rst.append(ki_rst)

    #计算不同lamb下的训练误差和方差
    dfs = np.zeros(len(lamb_lst))
    mse_means = np.zeros(len(lamb_lst))
    mse_mean_stds = np.zeros(len(lamb_lst))

    for lid in range(len(lamb_lst)):
        #lambda[lid]下的df
        dfs[lid] = train_mid_rst[0][lid][2]
        #K折CV下误差均值和标准差
        test_rsses = np.array(map(lambda i:train_mid_rst[i][lid][4], range(0,K)))
        mse_means[lid] = test_rsses.mean()
        #!!!!!注意：这儿求的是估计的标准差 1/K mean(sum(Xi)),  故而要除以K
        mse_mean_stds[lid] = test_rsses.std()/np.sqrt(K)

    best_lamb_id, minid = one_std_error_rule(mse_means, mse_mean_stds)
    print "Best lambid: %d, lambda: %.4f, degree of free: %.4f" % (best_lamb_id, lamb_lst[best_lamb_id], dfs[best_lamb_id]) 
    
    one_std_val = mse_means[minid] + mse_mean_stds[minid]
    plt.plot((dfs[0],dfs[-1]), (one_std_val, one_std_val), 'r-')
    plt.errorbar(dfs, mse_means, yerr=mse_mean_stds, fmt='-o')
    plt.savefig('images/mse_errorbar.png', format='png')

    #用K折选出来的最优lambda进行回归预测

if __name__ == '__main__':
    print "leasq by L2:"
    leasq_with_L2_new()
    print ""
