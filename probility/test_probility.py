#encoding: utf8
# test random

'''
第一章的测试程序
用伪随机数测试概率分布

从结果看，p=0.3 时，随试验次数增加，基本稳定在0.3 +- 0.005 左右， 
         p=0.03时，随试验次数增加，偏差还是比较大 0.03 +- 0.005
即：概率越小，随试验次数增加，偏差还是不小

'''

import random as rand
import matplotlib.pyplot as plt

def test_p(p, exp_count):
    match_count = 0
    
    max_id = 1000
    match_id = max_id * p
    total = 0.0
    rst = []
    
    for i in range(exp_count):
        cur_id = rand.randint(1, 1000)
        if cur_id <= match_id:
            match_count += 1
            total += cur_id
            rst.append(match_count*1.0/(i+1))
            
    return match_count * 1.0 / exp_count, rst
    

p = 0.3
ec = 1000

ep,rst = test_p(p, ec)

plt.plot(rst, 'bo')
plt.savefig('images/random_dist.png', format='png')
print "p: %s; exp_count:%s; match_ratio: %.3f" % (p, ec, ep)

p = 0.03
ec = 10000

ep,rst = test_p(p, ec)
print "p: %s; exp_count:%s; match_ratio: %.3f" % (p, ec, ep)

plt.clf()
plt.plot(rst, 'bo')
plt.savefig('images/random_dist2.png', format='png')
