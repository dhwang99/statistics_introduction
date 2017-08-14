#encoding: utf8

import numpy as np
from scipy.stats import norm
from scipy.stats import uniform 
import matplotlib.pyplot as plt
import pdb

from test_confidence_band import cal_and_plot_confidence_band

faithful_eruption = np.loadtxt('faithful.dat.lst')
etime = faithful_eruption[:,2]
eruption = faithful_eruption[:, 1]

hat_x_e = faithful_eruption.mean()
hat_es_e = faithful_eruption.std()
#pdb.set_trace()

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

