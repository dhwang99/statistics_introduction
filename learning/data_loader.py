#encoding: utf8

import numpy as np

'''
load data from file
return:
    train src data, test src data, normed training src dtaa, normed test src data, data_convert_fun
'''
def data_convert(val):
    if val == 'T':
        return 2
    elif val == 'F':
        return 1
    return 0

'''
问题1：
1. 关于标准化，是每一个因子在全集上标准化，还只是在测试集上标标准化？
我用的是在全集上标准化，这个应该更合理一些
全集上标准化

问题2：
2. 在求协方差时，np.cov的算法：sum(Xi-X_mean)(Yi-Y_mean)/(len(samples)-1)
   在实际上我算的时候，并没有减1. 这两个是不是第1个更准(无偏)但实际上用的时候是第2个?
   mean_data.T * mean_data / n

type:
    0:原始数据
    1:均值化
    2:标准化
need_bias: 是否需要偏置项(1)
y_standard: 是否对结果进行标准化
'''
def load_data(filename='data/prostate.data.txt', type=2, need_bias=1, y_standard=0):
    usecolums = range(1, 11)
    alldata = np.loadtxt(filename, skiprows=1, usecols=usecolums, converters={10:data_convert})
    t_indicate = np.where(alldata[:,9] == 2)
    f_indicate = np.where(alldata[:,9] == 1)

    train_X = np.take(alldata[:, 0:8], t_indicate[0], axis=0)
    test_X = np.take(alldata[:, 0:8], f_indicate[0], axis=0)

    train_Y = np.take(alldata[:,8], t_indicate[0], axis=0)
    test_Y = np.take(alldata[:,8], f_indicate[0], axis=0)

    return normalize_data(train_X, train_Y, test_X, test_Y, type, need_bias, y_standard)

'''
对数据标准化
type:
    0:原始数据
    1:均值化
    2:标准化
need_bias: 是否需要偏置项(1)
y_standard: 是否对结果进行标准化
'''
def normalize_data(train_X, train_Y, test_X, test_Y, type=2, need_bias=1, y_standard=0):
    X_stand_fun = lambda X:X
    Y_stand_fun = lambda Y:Y 
    if type > 0:
        mean_val = train_X.mean(axis=0)
        train_X = train_X - mean_val
        test_X = test_X - mean_val

        if type == 2:
            std_val = train_X.std(axis=0)
            std_val[std_val<=1e-5] = 1
            train_X = train_X / std_val
            test_X = test_X / std_val
            f_stand_fun = lambda X:(X - mean_val)/std_val
        else:
            f_stand_fun = lambda X:(X - mean_val)

    if need_bias > 0:
        train_X = np.hstack((np.ones((len(train_X), 1)), train_X))
        test_X = np.hstack((np.ones((len(test_X), 1)), test_X))

    if y_standard == 1:
        mean_val = train_Y.mean()
        train_Y = train_Y - mean_val
        test_Y = test_Y - mean_val

        Y_stand_fun = lambda y:y-mean_val

    return train_X, train_Y, test_X, test_Y, (X_stand_fun, Y_stand_fun)
