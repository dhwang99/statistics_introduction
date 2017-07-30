# encoding: utf8

'''
下面的基于U(0,1)实现了几个常用的连续随机变量概率分布的随机数模拟
U(0,1) 在0,1区间的连续均匀分布的随机数模拟

CDF: cumulate distribution function
PMF: probility mass function
PDF: probility density function

对任意随机变量X, 它有连续的的CDF F(x), 定义随机变量 Y=F(X), 则Y为 [0,1]上的均匀分布，即有 P(Y<=y) = y 
'''

import numpy as np
import matplotlib.pyplot as plt
import bisect
import pdb

import standard_normal_dist 

'''
根据标准正态分布表生成gauss分布。即：基于面积生成
X = Z*sigma + mu
'''
def gen_gauss_distribute_pdf_from_dist_table(mu, sigma, color='bo'):
    nt = standard_normal_dist.normal_dist_table
    std_x = list(standard_normal_dist.normal_dist_table_x)
    gauss_x = np.zeros(len(std_x))
    for id in range(len(std_x)):
        #Z = (X - mu)/sigma, X = Z * sigma + mu
        x = std_x[id] * sigma + mu 
        gauss_x[id] = x
    gauss_y = np.zeros(len(gauss_x))

    step = (gauss_x[-1] - gauss_x[0]) / len(gauss_x)
    pre_p = 0
    id = 0
    for x,p in nt:
        y = (p - pre_p) / step 
        pre_p = p
        gauss_y[id] = y
        id += 1

    return gauss_x, gauss_y

'''
生成高斯分布的x及其对应的pdf
f = 1/(sqrt(2*pi) * sigma) * exp(-(x-mu)^2/(2*sigma^2))
'''
def gen_gauss_distribute_pdf(mu, sigma, sample_count, max_x):
    gauss_x = np.linspace(-max_x, max_x, sample_count)
    gauss_y = map(lambda x:1/(np.sqrt(2 * np.pi) * sigma) * np.exp(-(x-mu)**2/(2*sigma**2)), gauss_x)

    return gauss_x, gauss_y

#画直方图
#a, 样本值，
def plt_hist(a, color='r', normed=False):
    sample_count = len(a)
    a.sort()
    group_count = int(np.sqrt(sample_count) + 0.9999) 
    width = (a[-1] - a[0]) / group_count

    #直方图间隔
    bins = np.linspace(a[0], a[-1], group_count+1)

    hist = np.zeros(group_count)
    for x in a:
        index = bisect.bisect_right(bins - 0.00001, x)   #减去最小的测量单位的一半，因为是浮点数，取个小的数
        #边界上的数归到上一个区间. 即x in [begin, end), 最后一个区间除外
        # 一般下界为 起始值 - 最小测量单位 * 0.5. 这样包含了最小元素，但上界值归到下一个区间了
        index -= 1
        if index >= group_count:
            index -= 1 #边界上的数。真实测量上不会发生

        hist[index] += 1

    if normed == True:
        total = np.sum(hist) * 1.0
        hist /= total
        hist /= width #概率是面积。除以这个得到高度

    plt.bar(bins[0:group_count], hist, width = width, facecolor = 'lightskyblue',edgecolor = 'white', align='edge')

    #也可以直接用这个来汇制直方图
    #plt.hist(a, bins = group_count, normed=normed)

#样本数一般要 > 100
#使用均匀分布随机数，生成符合正态分布的随机数
def gen_gauss_samples_byU(sample_count):
    sim_gauss_num = np.zeros(sample_count)
    sim_u = np.zeros(sample_count)
    
    for i in range(0, sample_count):
        u = np.random.uniform(0, 1)
        x = standard_normal_dist.guess_x(u)
        sim_gauss_num[i] = x
        sim_u[i] = u

    return sim_u, sim_gauss_num

'''
使用均匀分布随机数，生成符合指数分布的随机数
F(x) = 1 - exp(-1/beta*x)
x = -1/beta * 1/log(1 - F)     
'''
def gen_exp_samples_byU(sample_count, beta):
    sim_num = np.zeros(sample_count)
    sim_u = np.zeros(sample_count)
     
    for i in range(0, sample_count):
        u = np.random.uniform(0, 1)
        if u == 1:
            x = 0.
        else:
            x = -1./beta * np.log(1. - u)

        sim_u[i] = u
        sim_gauss_num[i] = x

    return sim_u, sim_gauss_num


'''
pdf = 1/beta * exp(-x/beta)
'''
def gen_exp_distribute_pdf(sample_count, beta, max_x):
    x = np.linspace(0, max_x, sample_count) 
    y = map(lambda k:1/beta * np.exp(-k/beta), x)

    return x,y

#分组的数量在5－12之间较为适宜(本程序用的是下一条)
#一般取总样本数开平方取上界. 上一个分组算法一般认为不合理
#分得越细，和曲线偏差越大,但好看；分得粗，不好看
sample_count = 1000
group_count = int(np.sqrt(sample_count) + 0.9999) 

sim_u, sim_gauss_num = gen_gauss_samples_byU(sample_count)
mean_val = np.mean(sim_gauss_num)
std_val = np.std(sim_gauss_num)
print "mean, std:", mean_val, std_val

#绘制高斯采样的直方图
plt_hist(sim_gauss_num, normed=True)
#叠加均匀分布采样直方图
plt.hist((sim_u-0.5)*8, bins = group_count, normed=True, color='r', alpha=0.6)

#叠加正态分布图
gauss_x, gauss_y = gen_gauss_distribute_pdf(0, 1, sample_count, np.max(sim_gauss_num))
plt.plot(gauss_x, gauss_y, color='g')

plt.savefig('images/norm_distribute_gen_by_U.png', format='png')

#绘制正态分布图.两种生成方法，对比一下误差
plt.clf()
gauss_x2, gauss_y2 = gen_gauss_distribute_pdf_from_dist_table(0, 1)
plt.plot(gauss_x, gauss_y, 'g-.')
plt.plot(gauss_x2, gauss_y2, 'r--')
gauss_x2, gauss_y2 = gen_gauss_distribute_pdf_from_dist_table(0, 2)
plt.plot(gauss_x2, gauss_y2, 'b-')
gauss_x2, gauss_y2 = gen_gauss_distribute_pdf_from_dist_table(0, 4)
plt.plot(gauss_x2, gauss_y2, 'y-')
plt.savefig('images/norm_distribute.png', format='png')

#指数分布相关
plt.clf()
sim_u, sim_num = gen_exp_samples_byU(sample_count, 1)
#绘制指数采样直方图
plt.hist(sim_num, bins = group_count, normed=True)
#叠加均匀分布采样直方图
plt.hist(sim_u*8, bins = group_count, normed=True, color='r', alpha=0.6)

#叠加标准指数分布图
exp_x, exp_y = gen_exp_distribute_pdf(sample_count, 1, np.max(sim_num))
plt.plot(exp_x, exp_y, color='g')

plt.savefig('images/norm_exp_gen_by_U.png', format='png')
