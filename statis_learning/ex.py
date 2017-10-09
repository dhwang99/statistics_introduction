#encoding: utf8

import numpy as np
import pdb

'''
load data from file
return:
    train src data, test src data, normed training src dtaa, normed test src data
'''
def data_convert(val):
    if val == 'T':
        return 1
    elif val == 'F':
        return 2
    return 0

'''
问题1：
1. 关于标准化，是每一个因子在全集上标准化，还只是在测试集上标标准化？
我用的是在全集上标准化，这个应该更合理一些
问题2：
2. 在求协方差时，np.cov的算法：sum(Xi-X_mean)(Yi-Y_mean)/(len(samples)-1)
   在实际上我算的时候，并没有减1. 这两个是不是第1个更准(无偏)但实际上用的时候是第2个?
   mean_data.T * mean_data / n

'''
def load_data(filename='data/prostate.data.txt'):
    usecolums = range(1, 11)
    alldata = np.loadtxt(filename, skiprows=1, usecols=usecolums, converters={10:data_convert})
    mean_val = alldata.mean(axis=0)
    mean_data = np.copy(alldata)
    mean_data = (mean_data[:,0:9] - mean_val[0:9])
    pdb.set_trace()
    mean_data = mean_data / mean_val[0:9]

    t_indicate = np.where(alldata[:,9] == 1)
    f_indicate = np.where(alldata[:,9] == 2)
    
    t_data = np.take(alldata, t_indicate[0], axis=0)[:,0:9]
    f_data = np.take(alldata, f_indicate[0], axis=0)[:,0:9]

    t_mean_data = np.take(mean_data, t_indicate[0], axis=0)
    f_mean_data = np.take(mean_data, f_indicate[0], axis=0)

    return t_data, f_data, t_mean_data, f_mean_data
     

if __name__ == '__main__':
    td, fd, tmd, fmd = load_data()
