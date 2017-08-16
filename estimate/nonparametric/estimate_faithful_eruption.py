#encoding: utf8

import numpy as np
from scipy.stats import norm
from scipy.stats import uniform 
import matplotlib.pyplot as plt
import pdb

from cdf_estimate import cal_and_plot_confidence_band

faithful_eruption = np.loadtxt('faithful.dat.lst')
etime = faithful_eruption[:,2]
eruption = faithful_eruption[:, 1]

hat_x_e = faithful_eruption.mean()
hat_es_e = faithful_eruption.std()
#pdb.set_trace()

alpha = 1 - 0.9
z_2a = 1.65
N = len(etime)
s = etime - hat_x_e
hat_var_of_X = 1./N * np.dot(s.T, s)   # variance of eruption time, not mean of eruption time. 样本方差
hat_se = np.sqrt(hat_var_of_X)/np.sqrt(N)  #the standard error of mean value of eruption time. 样本均值的标准差

confidence_interval = np.array([hat_x_e - z_2a * hat_se, hat_x_e + z_2a * hat_se])

print "%s confidence interval for mean is: %s" % (1-alpha, confidence_interval)

pdb.set_trace()

sorted_data = np.sort(etime)
N = sorted_data.shape[0]
alpha = 0.05

'''
以下用正态分布、均匀分布模拟，误差都比较大. 当下不知道用什么函数模拟比较合适. 后继估计会找到合适的函数

'''
stat = norm
cdf = stat.cdf(sorted_data, loc=hat_x_e, scale=hat_es_e)
cal_and_plot_confidence_band(sorted_data, cdf)
plt.savefig('images/failthful_eruption_estimate_by_%s.png'%stat.name, format='png')

plt.clf()
stat = uniform
cdf = stat.cdf(sorted_data, loc=sorted_data[0], scale=sorted_data[-1]-sorted_data[0])
cal_and_plot_confidence_band(sorted_data, cdf)
plt.savefig('images/failthful_eruption_estimate_by_%s.png'%stat.name, format='png')
