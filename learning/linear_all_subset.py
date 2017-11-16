#encoding: utf8

import numpy as np
import pdb
from scipy.stats import f as f_stats
import matplotlib.pyplot as plt

from data_loader import load_data, normalize_data
from linear_regression import leasq
from cv_common import gen_CV_samples_by_K_folds, one_std_error_rule

'''
子集选择
穷举法找最优模型 
注意：要求传入数据有偏置，这样可以算出截距情况, 不需要对Y进行标准化(这个教材上的不同)
'''
import itertools
def all_subset_regression(X, Y, X_t, Y_t, comp_testmse=1):
    beta_count = X.shape[1]
    beta_id_lst = range(1, beta_count)
    subids = range(0, beta_count)
    #所有的计算结果， 2**beta_count. 结构为: [sid][sub train_mse test_mse]
    subs = []
    #相同特征数下，最优的子集
    best_subs = []

    for sid in subids:
        subs.append([])
        #从给定的特征id列表beta_id_lst里，取出sid个的全部组合
        sub_list = list(itertools.combinations(beta_id_lst, sid))
        train_mse = 1e10
        test_mse = 1e10
        test_sub = []
        for sub in sub_list:
            #带了截距项
            sub = np.hstack((np.array([0]), np.array(sub, dtype='int')))
            _X= X[:, sub]
            _X_t = X_t[:, sub]
            beta, z_scores, _tre, _testmse, sig2_hat, se_beta_hat = leasq(_X, Y, _X_t, Y_t, sub)
            subs[sid].append((sub, _tre, _testmse))
            #取当前子集下，最小测试误差结果
            if (comp_testmse == 1 and (test_mse > _testmse)) or ((comp_testmse == 0) and (train_mse > _tre)):
                train_mse =_tre 
                test_mse = _testmse 
                test_sub = sub

        best_subs.append((test_sub, train_mse, test_mse))


    return subs, best_subs 

'''
用AIC选择最优特征
也就是选择每个p下，最小AIC对应的特征集合，并保存AIC值
其实和全算一遍AIC，然后取最小的方法一样

选出最小的后，用对应的特征进行回归训练和预测
AIC选择
AIC(M) = -2L(M) + 2p \sim RSS(M) + 2*sigma_hat^2 * p \sim RSS(M)/sigma_hat^2 + 2p

强调!!!!!!!
RSS是 trainX的 RSS !!!!!!!!
'''
from linear_regression import leasq

def all_subset_regression_by_AIC():
    #读取数据，并进行标准化
    X, Y, X_t, Y_t, dt_conv_fun = load_data()
    
    #求全局的训练sigma估计
    beta, z_scores, train_mse, test_mse, sigma2_hat, se_beta_hat = leasq(X, Y, X_t, Y_t)

    #计算所有子集选择下的结果. 注意下，比较的是train_mse
    subs, best_subs = all_subset_regression(X, Y, X_t, Y_t, comp_testmse=0)
    unzip_best_subs = zip(*best_subs)
    subids = range(0, len(best_subs))

    #求各最优子集下的AIC. 参数相同，则aic只与RSS相关，计算最优子集的RSS一定是最小的
    
    best_sub_AICs = map(lambda p:(unzip_best_subs[1][p]*len(Y)/sigma2_hat+2*p), subids)
    
    for bid in subids:
        print "by AIC rule:sub_id: %d, train mse: %.4f, test mse: %.4f, AIC: %.4f, subids: %s" % \
            (bid, unzip_best_subs[1][bid], unzip_best_subs[2][bid], best_sub_AICs[bid], unzip_best_subs[0][bid])
    
    #求AIC规则下的最优subid, best sub id
    print ""
    bid = np.argmin(best_sub_AICs)
    print "Best sub by AIC rule:best_sub_id: %d, train mse: %.4f, test mse: %.4f, AIC: %.4f, subids: %s" % \
            (bid, unzip_best_subs[1][bid], unzip_best_subs[2][bid], best_sub_AICs[bid], unzip_best_subs[0][bid])
   
    #绘制AIC随子集个数的变化图
    plt.clf()
    l2 = plt.plot(subids, best_sub_AICs, color='g', label='mse test')
    plt.legend(loc='best')
    plt.savefig('images/all_subset_regression_AIC.png', format='png')

    #绘制mse随子集个数的变化图. 这个和子集选择的相同
    plt.clf()
    l1 = plt.plot(subids, unzip_best_subs[1], color='r', label='mse train')
    l2 = plt.plot(subids, unzip_best_subs[2], color='g', label='mse test')
    plt.legend(loc='best')
    plt.savefig('images/all_subset_regression_AIC_mse.png', format='png')

    subs = np.array(unzip_best_subs[0][bid])
    beta, z_scores, _tre, _testmse, sig2_hat, se_beta_hat = \
            leasq(X[:,subs], Y, X_t[:, subs], Y_t)
    print "Best_rst: train_mse: %.4f  test_mse: %.4f" % (_tre, _testmse)


    return

'''
交叉验证选最佳子集
!!注：需要在相同的一组特征上应用CV，故而给的例子看起来是有问题的!
'''
def all_subset_regression_by_CV(same_to_book=False):
    #读取数据，不进行标准化, 由交叉验证进行标准化
    train_X, train_Y, test_X, test_Y, dt_conv_fun = load_data(type=0, need_bias=0, y_standard=0)
    K = 10
    CV_samples = gen_CV_samples_by_K_folds(train_X, train_Y, K)
    Kfold_subs = []
    for ki in range(K):
        cv_s = CV_samples[ki]
        train_X_CV, train_Y_CV, test_X_CV, test_Y_CV, nf = normalize_data(cv_s[0], cv_s[1], cv_s[2], cv_s[3])
        subs, best_subs = all_subset_regression(train_X_CV, train_Y_CV, test_X_CV, test_Y_CV)
        lin_subs = []
        if same_to_book == False:
            #对子集回归结果进行展开，方便计算
            map(lambda x:lin_subs.extend(x), subs)
        else:
            #和教材的算法一致
            lin_subs = best_subs

        Kfold_subs.append(lin_subs)

    subset_size = len(Kfold_subs[0])
    subset_testcv = np.zeros(subset_size)
    subset_testse = np.zeros(subset_size)
    subset_sids = []
    for i in range(subset_size):
        sub_i = zip(*map(lambda k:Kfold_subs[k][i], range(K)))
        
        sub_ids = sub_i[0][0]
        train_mse_cv = np.array(sub_i[1])
        test_mse_cv = np.array(sub_i[2])
        
        subset_sids.append(sub_ids)
        subset_testcv[i] = test_mse_cv.mean()
        subset_testse[i] = test_mse_cv.std()/np.sqrt(K)

    best_subid, min_id = one_std_error_rule(subset_testcv, subset_testse, reverse=True)
    #pdb.set_trace()

    print "best_subid: %d, best_subs: %s" % (best_subid, subset_sids[best_subid])

    X, Y, X_t, Y_t, dt_conv_fun = load_data()
    subs = np.array(subset_sids[best_subid])
    beta, z_scores, _tre, _testmse, sig2_hat, se_beta_hat = \
            leasq(X[:,subs], Y, X_t[:, subs], Y_t)
    print "Best_rst: train_mse: %.4f  test_mse: %.4f" % (_tre, _testmse)

    return

if __name__ == '__main__':
    #读取数据，并进行标准化
    X, Y, X_t, Y_t, dt_cf = load_data(need_bias=0, y_standard=1)
    subs, best_subs = all_subset_regression(X, Y, X_t, Y_t)
    unzip_best_subs = zip(*best_subs)

    subids = range(0, len(best_subs))

    "all subset regression:"
    #输出不同特征数下的最优子集
    #根据 test_mse 结果进行子集选择(选最优的即可，和k折CV不同：CV有一个一倍标准差准则)
    best_subid = 0
    test_mse = 1e10

    for i in subids:
        if best_subs[i][2] < test_mse:
            test_mse = best_subs[i][2]
            best_subid = i
        print "sid:%d train_mse:%.4f mse_tst:%.4f subset:%s" % (i, best_subs[i][1], best_subs[i][2], best_subs[i][0])

    #输出最优子集
    print "best subid is: %d, features are: %s, train mse: %.4f test mse: %.4f" % \
            (best_subid, best_subs[best_subid][0], best_subs[best_subid][1], best_subs[best_subid][2])
    
    l1 = plt.plot(subids, unzip_best_subs[1], color='r', label='mse train')
    l2 = plt.plot(subids, unzip_best_subs[2], color='g', label='mse test')
    plt.legend(loc='best')
    plt.savefig('images/all_subset_regression.png', format='png')
    plt.clf()
    print ""

    #子集选择: AIC评价
    print "all_subset_regression_AIC:"
    all_subset_regression_by_AIC()

    '''
    重复了10次，从结果看，书上给的算法是很不稳定的
    '''
    for i in range(2):
        plt.clf()
        print ""
        #子集选择: CV评价
        print "cross validation, same_to_book=False"
        all_subset_regression_by_CV()
    
        #子集选择: CV评价, 和教材一致
        print "cross validation, same_to_book=True"
        all_subset_regression_by_CV(same_to_book=True)
